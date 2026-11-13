_MRI_MODEL_SCOPE='compact'

from pathlib import Path
import time,json,hashlib,gc,copy
import numpy as np
import pandas as pd
import joblib
from scipy.ndimage import zoom
from scipy.special import expit
from scipy.stats import beta
from sklearn.base import clone,BaseEstimator,TransformerMixin
from sklearn.feature_selection import SelectPercentile,f_classif
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,confusion_matrix,
 precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,
 matthews_corrcoef,brier_score_loss,log_loss,roc_curve,precision_recall_curve)

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)))

def guard(cfg):
    if time.time()>cfg['deadline_unix']:
        raise TimeoutError('Compute budget reached. Completed fits are cached. Resume with the same configuration and output folder.')

def grids(name):
    if name=='mri' and _MRI_MODEL_SCOPE=='compact':
        return {'Voxel_LR':[dict(C=c) for c in (.01,.1,1.)],
                'Voxel_SVM':[dict(C=c) for c in (.1,1.)],
                'MedicalNet_Global_LR':[dict(C=c,percentile=100) for c in (.01,.1,1.)],
                'MedicalNet_Regional_LR':[dict(C=c,percentile=100) for c in (.01,.1,1.)]}
    if name in ['voice','spiral']:
        base={
            'LogisticRegression':[dict(C=c,percentile=p) for c in (.1,1.,10.) for p in (50,100)],
            'RBF_SVM':[dict(C=c,percentile=p) for c in (1.,10.) for p in (50,100)],
            'RandomForest':[dict(max_depth=d) for d in (3,5)],
            'GradientBoosting':[dict(max_depth=d,learning_rate=.05) for d in (1,2)],
            'SoftVoting':[dict(C=1.)]}
        if name=='spiral':
            base={family:[dict(param,columns=cols,task_mode=mode) for param in params for mode,cols in [('both',None),('static',list(range(6))),('dynamic',list(range(6,12)))]] for family,params in base.items()}
        return base
    standard = {
        'Voxel_SVM':[dict(C=c) for c in (.1,1.)],
        'Frozen_ResNet_LR':[dict(C=c) for c in (.1,1.)],
        'Frozen_ResNet_SVM':[dict(C=c) for c in (.1,1.)],
        'Spatial_LR':[dict(C=c,percentile=p) for c in (.1,1.,10.) for p in (50,100)],
        'Spatial_SVM':[dict(C=c,percentile=p) for c in (1.,10.) for p in (50,100)],
        'FineTune_ResNet18':[dict(lr=.0001)]}
    standard.update({family:[dict(C=c,percentile=p) for c in (.01,.1,1.) for p in (50,100)] for family in ['MedicalNet_Global_LR','MedicalNet_Global_SVM','MedicalNet_Regional_LR','MedicalNet_Regional_SVM']})
    return standard

def make_pipeline(family,param,seed):
    if family.startswith('MedicalNet_'):family='Frozen_ResNet_LR' if family.endswith('_LR') else 'Frozen_ResNet_SVM'
    if family in ['LogisticRegression','Frozen_ResNet_LR','Spatial_LR','Voxel_LR']:
        model=LogisticRegression(C=param.get('C',1.),max_iter=4000,random_state=seed)
    elif family in ['RBF_SVM','Voxel_SVM','Frozen_ResNet_SVM','Spatial_SVM']:
        model=SVC(C=param.get('C',1.),kernel='rbf',gamma='scale',probability=False,random_state=seed)
    elif family=='RandomForest':
        model=RandomForestClassifier(n_estimators=200,max_depth=param['max_depth'],min_samples_leaf=2,n_jobs=2,random_state=seed)
    elif family=='GradientBoosting':
        model=GradientBoostingClassifier(n_estimators=100,max_depth=param['max_depth'],learning_rate=param['learning_rate'],random_state=seed)
    elif family=='SoftVoting':
        model=VotingClassifier(estimators=[('lr',LogisticRegression(C=param.get('C',1.),max_iter=4000,random_state=seed)),('rf',RandomForestClassifier(n_estimators=200,max_depth=5,min_samples_leaf=2,n_jobs=2,random_state=seed)),('gb',GradientBoostingClassifier(n_estimators=100,max_depth=1,learning_rate=.05,random_state=seed))],voting='soft',n_jobs=1)
    else:raise ValueError('Unknown family '+family)
    return Pipeline([
        ('columns',FeatureColumns(param.get('columns'))),
        ('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
        ('scale',StandardScaler()),
        ('selection',SelectPercentile(safe_f_classif,percentile=param.get('percentile',100))),
        ('pca',PCA(n_components=.95,svd_solver='full') if family.startswith(('Voxel_','Frozen_')) else 'passthrough'),
        ('model',model)])

def model_score(model,X):
    classes=np.asarray(model.classes_)
    if not np.array_equal(classes,[0,1]):raise ValueError('Expected classes [0, 1]; PD=1.')
    if hasattr(model,'predict_proba'):
        return model.predict_proba(X)[:,int(np.flatnonzero(classes==1)[0])]
    # বাংলা: sigmoid শুধু bounded score; calibrated probability দাবি করা হবে না।
    return expit(model.decision_function(X))

def voxel_vector(volume,shape=(12,14,12)):
    a=np.asarray(volume,np.float32)
    assert a.ndim==3 and np.isfinite(a).all()
    b=zoom(a,np.array(shape)/np.array(a.shape),order=1,prefilter=False)
    assert b.shape==tuple(shape)
    return b.ravel().astype(np.float32)

class FineTuneMRI:
    """Patient-level multi-view fine-tuning with train-only early stopping and refit."""
    def __init__(self,checkpoint,cfg,lr=.0001,seed=42):
        self.checkpoint=str(checkpoint);self.cfg=dict(cfg);self.lr=lr;self.seed=seed
        self.classes_=np.array([0,1]);self.history_=[];self.selected_epochs_=None
    def _init(self):
        import torch
        from torchvision.models import resnet18
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(self.seed)
        self.encoder=resnet18(weights=None);self.encoder.fc=torch.nn.Identity()
        self.encoder.load_state_dict(torch.load(self.checkpoint,map_location='cpu',weights_only=True))
        for p in self.encoder.parameters():p.requires_grad=False
        for p in self.encoder.layer4.parameters():p.requires_grad=True
        self.head=torch.nn.Sequential(torch.nn.Dropout(.3),torch.nn.Linear(512,1))
    def _batch(self,paths,device,training=False):
        import torch
        import torch.nn.functional as F
        xs=[]
        for path in paths:
            a=np.load(str(path),allow_pickle=False)
            # বাংলা: সব view/slice একই patient-এর; একই split-এ থাকবে।
            x=slice_images(a,self.cfg['cnn_slices_per_view'])
            x=F.interpolate(x,size=(160,160),mode='bilinear',align_corners=False)
            xs.append(x)
        return torch.stack(xs).to(device)
    def _forward(self,x):
        n,s,c,h,w=x.shape
        z=self.encoder(x.reshape(n*s,c,h,w)).reshape(n,s,-1).mean(dim=1)
        return self.head(z).flatten()
    def _train(self,paths,y,epochs,val_paths=None,val_y=None,early=False):
        import torch
        device='cuda' if torch.cuda.is_available() else 'cpu'
        # CPU fallback is valid; expected to be much slower than GPU.
        self.encoder.to(device);self.head.to(device)
        params=[p for p in self.encoder.parameters() if p.requires_grad]+list(self.head.parameters())
        optimizer=torch.optim.AdamW(params,lr=self.lr,weight_decay=.001)
        n0=max(1,int((y==0).sum()));n1=max(1,int((y==1).sum()))
        criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n0/n1,device=device))
        rng=np.random.default_rng(self.seed);best=float('inf');bad=0;selected=epochs
        for epoch in range(1,epochs+1):
            guard(self.cfg);self.encoder.eval();self.encoder.layer4.train();self.head.train()
            # বাংলা: অল্প patient-এ BatchNorm running statistics freeze থাকবে।
            for layer in self.encoder.modules():
                if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.eval()
            order=rng.permutation(len(paths));losses=[]
            for k in range(0,len(order),2):
                guard(self.cfg);idx=order[k:k+2];x=self._batch(paths[idx],device,True)
                target=torch.tensor(y[idx],dtype=torch.float32,device=device)
                optimizer.zero_grad(set_to_none=True);loss=criterion(self._forward(x),target)
                loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();losses.extend([float(loss.detach().cpu())]*len(idx))
            record={'epoch':epoch,'train_loss':float(np.mean(losses))}
            if val_paths is not None:
                self.encoder.eval();self.head.eval();v_losses=[]
                with torch.inference_mode():
                    for k in range(0,len(val_paths),2):
                        x=self._batch(val_paths[k:k+2],device)
                        target=torch.tensor(val_y[k:k+2],dtype=torch.float32,device=device)
                        v_losses.extend([float(criterion(self._forward(x),target).cpu())]*len(target))
                v=float(np.mean(v_losses));record['validation_loss']=v
                if v<best-1e-4:best=v;selected=epoch;bad=0
                else:bad+=1
            self.history_.append(record)
            print(f'    CNN epoch {epoch}/{epochs} | loss={record["train_loss"]:.4f}'+(f' | validation_loss={record["validation_loss"]:.4f}' if 'validation_loss' in record else ''),flush=True)
            if early and bad>=self.cfg['cnn_patience']:break
        self.encoder.cpu();self.head.cpu();torch.cuda.empty_cache();return selected
    def fit(self,paths,y):
        paths=np.asarray(paths,str);y=np.asarray(y,int)
        assert np.bincount(y,minlength=2).min()>=3,'Too few MRI training subjects for early stopping.'
        tr,va=train_test_split(np.arange(len(y)),test_size=max(2,int(round(len(y)*.25))),random_state=self.seed,stratify=y)
        self._init()
        self.selected_epochs_=self._train(paths[tr],y[tr],self.cfg['cnn_max_epochs'],paths[va],y[va],True)
        self.early_stopping_history_=list(self.history_)
        # বাংলা: epoch নির্ধারণের পরে সব fit-input patient দিয়ে পুনরায় train।
        self._init();self.history_=[]
        self._train(paths,y,self.selected_epochs_)
        self.refit_history_=list(self.history_);return self
    def predict_proba(self,paths):
        import torch
        device='cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder.to(device).eval();self.head.to(device).eval();scores=[]
        paths=np.asarray(paths,str)
        with torch.inference_mode():
            for k in range(0,len(paths),2):
                x=self._batch(paths[k:k+2],device);scores.extend(torch.sigmoid(self._forward(x)).cpu().numpy())
        self.encoder.cpu();self.head.cpu()
        if device=='cuda':torch.cuda.empty_cache()
        p=np.asarray(scores);return np.column_stack([1-p,p])

def training_input(spec,ids,family):
    ids=list(ids);frame=spec['subjects'].set_index('subject_id').loc[ids]
    if spec['name']=='voice':
        obs=spec['observations'][spec['observations'].subject_id.isin(ids)].copy()
        X=obs[spec['features']].to_numpy(float);y=obs.label.to_numpy(int)
        counts=obs.groupby('subject_id').size();class_counts=frame.label.value_counts()
        # বাংলা: বেশি recording থাকা patient অতিরিক্ত weight পাবে না।
        weights=np.array([1/(counts[s]*class_counts[l]) for s,l in zip(obs.subject_id,y)])
        weights/=weights.mean();return X,y,weights
    y=frame.label.to_numpy(int)
    if family=='FineTune_ResNet18':return frame.array_path.to_numpy(str),y,None
    columns=feature_columns(spec,family)
    X=frame[columns].to_numpy(float)
    counts=pd.Series(y).value_counts();weights=np.array([len(y)/(2*counts[l]) for l in y])
    return X,y,weights

def predict_subjects(spec,ids,bundle):
    ids=list(ids);family=bundle['family'];model=bundle['model']
    if spec['name']=='voice':
        obs=spec['observations'][spec['observations'].subject_id.isin(ids)].copy()
        obs['score']=model_score(model,obs[spec['features']].to_numpy(float))
        p=obs.groupby('subject_id').score.mean().reindex(ids).to_numpy()
    else:
        frame=spec['subjects'].set_index('subject_id').loc[ids]
        if family=='FineTune_ResNet18':p=model.predict_proba(frame.array_path.to_numpy(str))[:,1]
        else:
            cols=feature_columns(spec,family)
            p=model_score(model,frame[cols].to_numpy(float))
    assert np.isfinite(p).all() and len(p)==len(ids)
    return p

def fit_candidate(spec,ids,family,param,cfg,seed):
    # বাংলা: cache identity-তে training subjects, config ও source signature থাকবে।
    signature={k:v for k,v in cfg.items() if k not in ['deadline_unix','max_compute_hours']}
    X,y,weights=training_input(spec,ids,family)
    training_label_hash=hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()
    key=hashlib.sha256(json.dumps(dict(training_label_hash=training_label_hash,dataset=spec['signature'],features=spec.get('features'),voxel_features=spec.get('voxel_features'),spatial_features=spec.get('spatial_features'),ids=sorted(ids),family=family,param=param,cfg=signature,seed=seed),sort_keys=True).encode()).hexdigest()
    cache=Path(cfg['output'])/'fit_cache'/f'{key}.joblib'
    if cache.exists():
        cached=joblib.load(cache)
        assert set(cached['training_subjects'])==set(ids)
        return cached
    guard(cfg)
    if family=='FineTune_ResNet18':
        model=FineTuneMRI(spec['checkpoint'],cfg,param['lr'],seed).fit(X,y)
    else:
        model=make_pipeline(family,param,seed);model.fit(X,y,model__sample_weight=weights)
    bundle=dict(model=model,family=family,param=param,training_subjects=sorted(ids),dataset=spec['name'],fit_cache_key=key,seed=seed,score_calibrated=False)
    temporary=cache.with_suffix('.partial');joblib.dump(bundle,temporary);temporary.replace(cache)
    return bundle

def choose_key(record):
    return (record['inner_balanced_accuracy'],record['inner_roc_auc'],-abs(record.get('threshold',.5)-.5))

def tune_family(spec,train_ids,family,params,cfg,seed):
    frame=spec['subjects'].set_index('subject_id').loc[list(train_ids)].reset_index()
    folds=min(3,int(frame.label.value_counts().min()));assert folds>=2
    splits=list(StratifiedKFold(folds,shuffle=True,random_state=seed).split(frame,frame.label))
    thresholds=np.asarray(cfg.get('mri_threshold_grid',[.5]) if spec['name']=='mri' else cfg.get('threshold_grid',[.3,.35,.4,.45,.5,.55,.6,.65,.7]),float)
    if not len(thresholds) or not np.isfinite(thresholds).all() or np.any((thresholds<=0)|(thresholds>=1)):raise ValueError('Invalid prespecified threshold grid.')
    records=[]
    for param in params:
        parts=[];aucs=[]
        for k,(tr,va) in enumerate(splits):
            trids=frame.subject_id.iloc[tr].tolist();vaids=frame.subject_id.iloc[va].tolist()
            assert set(trids).isdisjoint(vaids)
            fit=fit_candidate(spec,trids,family,param,cfg,seed+k+100)
            p=predict_subjects(spec,vaids,fit);y=frame.label.iloc[va].to_numpy(int)
            parts.append((y,p));aucs.append(roc_auc_score(y,p))
        values=[float(np.mean([balanced_accuracy_score(y,p>=threshold) for y,p in parts])) for threshold in thresholds]
        index=max(range(len(thresholds)),key=lambda j:(values[j],-abs(thresholds[j]-.5)))
        records.append(dict(family=family,param=param,threshold=float(thresholds[index]),inner_balanced_accuracy=values[index],inner_roc_auc=float(np.mean(aucs)),threshold_candidates=thresholds.tolist(),inner_ba_by_threshold=values,interpretation='Joint parameter/threshold tuning; inner score optimistic, outer evaluation required'))
    return max(records,key=choose_key),records

def compare_development(spec,cfg):
    name=spec['name'];out=Path(cfg['output']);table=out/'tables';models=out/'models'
    dev=spec['subjects'].query("split=='development'").reset_index(drop=True)
    assert dev.subject_id.is_unique and set(dev.label)=={0,1}
    if dev.label.value_counts().min()<6:raise ValueError('Need >=6 development subjects per class for this nested protocol.')
    candidates=grids(name);fold_records=[];inner_records=[];outer_choices=[]
    scores={f:np.full(len(dev),np.nan) for f in candidates};cuts={f:np.full(len(dev),np.nan) for f in candidates}
    selected_p=np.full(len(dev),np.nan);selected_t=np.full(len(dev),np.nan)
    for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=cfg['seed']).split(dev,dev.label),1):
        trids=dev.subject_id.iloc[tr].tolist();vaids=dev.subject_id.iloc[va].tolist();winners=[]
        for family,params in candidates.items():
            print(f'{name.upper()} | outer {fold}/3 | {family} | feature/config/threshold tuning',flush=True)
            winner,records=tune_family(spec,trids,family,params,cfg,cfg['seed']+fold)
            inner_records.extend(dict(stage=f'outer_{fold}',**r) for r in records)
            bundle=fit_candidate(spec,trids,family,winner['param'],cfg,cfg['seed']+fold+500)
            p=predict_subjects(spec,vaids,bundle);t=winner['threshold'];scores[family][va]=p;cuts[family][va]=t
            m=score(dev.label.iloc[va],p,t);train_m=score(dev.label.iloc[tr],predict_subjects(spec,trids,bundle),t)
            fold_records.append(dict(fold=fold,family=family,selected_param=json.dumps(winner['param'],sort_keys=True),training_accuracy=train_m['accuracy'],training_balanced_accuracy=train_m['balanced_accuracy'],**m));winners.append(winner)
        choice=max(winners,key=choose_key);selected_p[va]=scores[choice['family']][va];selected_t[va]=choice['threshold'];outer_choices.append(dict(fold=fold,**choice))
        pd.DataFrame(fold_records).to_csv(table/f'{name}_candidate_outer_folds.csv',index=False)
    pd.DataFrame(inner_records).to_csv(table/f'{name}_inner_search.csv',index=False);write_json(table/f'{name}_outer_selection.json',outer_choices)
    comparison=[]
    for family in candidates:
        p=scores[family];t=cuts[family];assert np.isfinite(p).all() and np.isfinite(t).all()
        pred=dev[['subject_id','label']].copy();pred['probability_pd']=p;pred['decision_threshold']=t;pred['prediction_pd']=(p>=t).astype(int)
        pred.to_csv(table/f'{name}_{family}_oof.csv',index=False);comparison.append(dict(family=family,**score(dev.label,p,t)))
    comparison=pd.DataFrame(comparison).sort_values('balanced_accuracy',ascending=False)
    comparison.to_csv(table/f'{name}_four_model_comparison.csv',index=False)
    comparison.to_csv(table/f'{name}_candidate_comparison.csv',index=False)
    selected_oof=dev[['subject_id','label']].copy();selected_oof['probability_pd']=selected_p;selected_oof['decision_threshold']=selected_t;selected_oof['prediction_pd']=(selected_p>=selected_t).astype(int)
    assert np.isfinite(selected_p).all() and np.isfinite(selected_t).all()
    selected_oof.to_csv(table/f'{name}_selection_procedure_oof.csv',index=False)
    write_json(table/f'{name}_selection_procedure_metrics.json',score(dev.label,selected_p,selected_t))
    final_choices=[];records=[]
    for family,params in candidates.items():
        winner,search=tune_family(spec,dev.subject_id.tolist(),family,params,cfg,cfg['seed']+900)
        final_choices.append(winner);records.extend(search)
    final_choice=max(final_choices,key=choose_key)
    pd.DataFrame(records).to_csv(table/f'{name}_final_development_search.csv',index=False)
    final=fit_candidate(spec,dev.subject_id.tolist(),final_choice['family'],final_choice['param'],cfg,cfg['seed']+1000)
    final=dict(final,features=feature_columns(spec,final_choice['family']),threshold=final_choice['threshold'],score_calibrated=False)
    joblib.dump(final,models/f'{name}_selected_bundle.joblib')
    write_json(table/f'{name}_selection_lock.json',dict(**final_choice,criterion='Inner subject balanced accuracy then AUC; threshold tie closest to 0.5',test_used=False,protocol='Retrospective V7; previous test outcomes known'))
    print(comparison[['family','accuracy','balanced_accuracy','roc_auc','specificity']].round(4).to_string(index=False))
    print(name.upper(),'| LOCKED:',final_choice['family'],final_choice['param'],'| threshold:',final_choice['threshold'])
    return dict(bundle=final,oof=selected_oof,comparison=comparison,choice=final_choice)

def holdout_evaluate(spec,result,cfg):
    frame=spec['subjects'].query("split=='test'").reset_index(drop=True);bundle=result['bundle'];t=float(bundle['threshold'])
    assert set(frame.subject_id).isdisjoint(bundle['training_subjects'])
    pred=frame[['subject_id','label']].copy();p=predict_subjects(spec,frame.subject_id.tolist(),bundle)
    pred['probability_pd']=p;pred['decision_threshold']=t;pred['prediction_pd']=(p>=t).astype(int)
    m=score(pred.label,p,t);m['constant_predicted_class']=bool(pred.prediction_pd.nunique()==1);m['score_range']=float(np.ptp(p))
    pred.to_csv(Path(cfg['output'])/'tables'/f'{spec["name"]}_selected_retrospective_holdout.csv',index=False)
    write_json(Path(cfg['output'])/'tables'/f'{spec["name"]}_selected_retrospective_metrics.json',m)
    write_json(Path(cfg['output'])/'tables'/f'{spec["name"]}_fixed_05_holdout_diagnostic.json',dict(**score(pred.label,p,.5),interpretation='Secondary diagnostic only; does not select threshold'))
    print(spec['name'].upper(),'| retrospective holdout:',json.dumps(m));return pred,m


def score(y, p, threshold=0.5):
    y = np.asarray(y, int)
    raw=np.asarray(p,float)
    if y.ndim!=1 or raw.ndim!=1 or len(y)!=len(raw) or len(y)==0:raise ValueError('Nonempty equal-length 1D labels/scores required.')
    if not set(y)<= {0,1} or not np.isfinite(raw).all() or np.any((raw<0)|(raw>1)):raise ValueError('Invalid labels or scores.')
    p = np.clip(raw, 1e-07, 1 - 1e-07)
    cuts=np.asarray(threshold,float)
    if cuts.ndim>1 or (cuts.ndim==1 and cuts.shape!=y.shape) or not np.isfinite(cuts).all() or np.any((cuts<=0)|(cuts>=1)):raise ValueError('Invalid decision threshold(s).')
    pred = (raw >= cuts).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = len(y)
    k = int((pred == y).sum())
    return dict(n=n, accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred), sensitivity=tp / (tp + fn) if tp + fn else None, specificity=tn / (tn + fp) if tn + fp else None, precision=precision_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0), roc_auc=roc_auc_score(y, p) if len(set(y)) == 2 else None, average_precision=average_precision_score(y, p) if len(set(y)) == 2 else None, mcc=matthews_corrcoef(y, pred), brier=brier_score_loss(y, p), log_loss=log_loss(y, p, labels=[0, 1]), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), threshold=float(cuts) if cuts.ndim==0 else None, accuracy_ci_low=float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0, accuracy_ci_high=float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0)

def publication_plots(pred, title, stem, threshold=0.5):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    y = pred.label.to_numpy(int)
    p = pred.probability_pd.to_numpy(float)
    cuts=decision_thresholds(pred)
    decision=(p>=cuts).astype(int)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    fpr, tpr, _ = roc_curve(y, p)
    axes[0, 0].plot(fpr, tpr, label=f'AUC = {roc_auc_score(y, p):.3f}')
    axes[0, 0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0, 0].set(xlabel='False positive rate', ylabel='True positive rate', title='ROC')
    axes[0, 0].legend()
    precision, recall, _ = precision_recall_curve(y, p)
    axes[0, 1].step(recall, precision, where='post', label=f'AP = {average_precision_score(y, p):.3f}')
    axes[0, 1].axhline(y.mean(), ls='--', color='gray', label='Prevalence')
    axes[0, 1].set(xlabel='Recall', ylabel='Precision', title='Precision–recall')
    axes[0, 1].legend()
    prob_true, prob_pred = calibration_curve(y, p, n_bins=4, strategy='uniform')
    axes[0, 2].plot(prob_pred, prob_true, 'o-')
    axes[0, 2].plot([0, 1], [0, 1], '--', color='gray')
    axes[0, 2].set(xlabel='Mean predicted probability', ylabel='Observed PD fraction', title='Calibration (small sample)')
    ConfusionMatrixDisplay.from_predictions(y, decision, display_labels=['Control', 'PD'], ax=axes[1, 0], colorbar=False, cmap='Blues')
    axes[1, 0].set_title('Confusion matrix (development-selected thresholds)')
    for lab, color in [(0, '#2878b5'), (1, '#d9534f')]:
        axes[1, 1].hist(p[y == lab], bins=np.linspace(0, 1, 11), alpha=0.65, label=['Control', 'PD'][lab], color=color)
    axes[1, 1].set(xlabel='Predicted PD probability', ylabel='Subjects', title='Probability distribution')
    axes[1, 1].legend()
    ts = np.linspace(0.05, 0.95, 50)
    sens = []
    spec = []
    for t in ts:
        m = score(y, p, t)
        sens.append(m['sensitivity'])
        spec.append(m['specificity'])
    axes[1, 2].plot(ts, sens, label='Sensitivity')
    axes[1, 2].plot(ts, spec, label='Specificity')
    
    for cut in np.unique(cuts):axes[1, 2].axvline(cut,ls='--',color='gray',alpha=.4)
    axes[1, 2].set(xlabel='Threshold', ylabel='Rate', title='Threshold analysis (descriptive only)')
    axes[1, 2].legend()
    for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 2]]:
        ax.set_ylim(0, 1.03)
    fig.suptitle(title)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(str(stem) + '.' + ext, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def slice_images(volume, n_slices=8):
    import torch
    import torch.nn.functional as F
    a = np.asarray(volume, np.float32)
    assert a.ndim == 3 and np.isfinite(a).all()
    images = []
    for axis in range(3):
        for frac in np.linspace(0.25, 0.75, n_slices):
            i = int(round((a.shape[axis] - 1) * frac))
            s = np.take(a, i, axis=axis).copy()
            t = torch.from_numpy(s)[None, None]
            t = F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)[0].repeat(3, 1, 1)
            images.append(t)
    x = torch.stack(images)
    return (x - torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]) / torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]


def feature_columns(spec,family):
    if family.startswith('MedicalNet_Global_'):return spec['medical_global_features']
    if family.startswith('MedicalNet_Regional_'):return spec['medical_regional_features']
    if family in ['Voxel_SVM','Voxel_LR']:return spec['voxel_features']
    if family.startswith('Spatial_'):return spec['spatial_features']
    return spec['features']

def decision_thresholds(frame):
    return frame['decision_threshold'].to_numpy(float) if 'decision_threshold' in frame else np.full(len(frame),.5)

def predicted_labels(frame):
    return (frame.probability_pd.to_numpy(float)>=decision_thresholds(frame)).astype(int)

def score_frame(frame):
    return score(frame.label,frame.probability_pd,decision_thresholds(frame))


class FeatureColumns(BaseEstimator,TransformerMixin):
    def __init__(self,columns=None):self.columns=columns
    def fit(self,X,y=None):
        X=np.asarray(X);self.n_features_in_=X.shape[1]
        if self.columns is not None and (not len(self.columns) or len(set(self.columns))!=len(self.columns) or min(self.columns)<0 or max(self.columns)>=X.shape[1]):raise ValueError('Invalid prespecified feature subset.')
        return self
    def transform(self,X):
        X=np.asarray(X)
        if X.ndim!=2 or X.shape[1]!=self.n_features_in_:raise ValueError('Feature schema mismatch.')
        return X if self.columns is None else X[:,self.columns]

def safe_f_classif(X,y):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');scores,pvalues=f_classif(X,y)
    return np.nan_to_num(scores,nan=0.,posinf=np.finfo(float).max,neginf=0.),np.nan_to_num(pvalues,nan=1.)

def spatial_vector(volume,grid=(3,4,3)):
    """Atlas-free intensity/foreground grid summaries; NOT anatomical ROI volumes.
    Grid indices follow the trained z,y,x reference; foreground is intensity>0.
    Occupancy/contrast may encode acquisition, age/sex or skull-strip differences.
    """
    a=np.asarray(volume,np.float32)
    if a.ndim!=3 or not np.isfinite(a).all() or a.min()<0 or a.max()>1:raise ValueError('Expected finite normalized 3D MRI in [0,1].')
    mask=a>0
    if mask.sum()<100:raise ValueError('Insufficient MRI foreground.')
    gradients=np.sqrt(sum(g*g for g in np.gradient(a)))
    values=[]
    zblocks=np.array_split(np.arange(a.shape[0]),grid[0]);yblocks=np.array_split(np.arange(a.shape[1]),grid[1]);xblocks=np.array_split(np.arange(a.shape[2]),grid[2])
    for iz in zblocks:
        for iy in yblocks:
            for ix in xblocks:
                q=np.ix_(iz,iy,ix);v=a[q];m=mask[q];brain=v[m]
                values.append(float(m.mean()))
                if len(brain):values.extend([float(brain.mean()),float(brain.std()),*np.quantile(brain,[.25,.5,.75]).tolist(),float(gradients[q][m].mean())])
                else:values.extend([0.]*6)
    foreground=a[mask];values.extend([float(mask.mean()),float(foreground.mean()),float(foreground.std()),*np.quantile(foreground,[.05,.25,.5,.75,.95]).tolist()])
    left,right=np.array_split(a,2,axis=2);lm,rm=left>0,right>0
    values.extend([float(abs(lm.mean()-rm.mean())),float(abs(left[lm].mean()-right[rm].mean())) if lm.any() and rm.any() else 0.])
    result=np.asarray(values,np.float32)
    if not np.isfinite(result).all():raise ValueError('Invalid spatial MRI descriptor.')
    return result

def usable_input_rows(bundle,X):
    values=np.asarray(X,float)
    if values.ndim!=2:raise ValueError('Expected a 2D subject-feature matrix.')
    columns=bundle.get('param',{}).get('columns')
    if columns is not None:values=values[:,columns]
    return np.isfinite(values).any(axis=1)
