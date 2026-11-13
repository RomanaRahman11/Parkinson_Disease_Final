"""Segment-aware spiral features. Resampling is a sensitivity choice, not proof of deconfounding."""
import numpy as np
SHAPE_FEATURE_NAMES=['path_efficiency','step_cv','abs_turn_median','abs_turn_p90','radius_cv','turn_consistency']
def shape_features(frame,n_points=256):
    result={};quality=[]
    for task in (0,1):
        values={k:np.nan for k in SHAPE_FEATURE_NAMES}
        # Preserve original order; never join across another task or a backwards timestamp.
        ids=np.flatnonzero(frame.TestID.to_numpy(int)==task)
        blocks=[]
        if len(ids):
            ts=frame.Timestamp.to_numpy(float)
            cut=np.flatnonzero((np.diff(ids)!=1)|(np.diff(ts[ids])<0))+1
            for index in np.split(ids,cut):
                xy=frame.iloc[index][['X','Y']].to_numpy(float)
                if len(xy)<5:continue
                keep=np.r_[True,np.linalg.norm(np.diff(xy,axis=0),axis=1)>1e-10];xy=xy[keep]
                if len(xy)<5:continue
                arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
                if arc[-1]<=1e-8:continue
                grid=np.linspace(0,arc[-1],n_points)
                xy=np.column_stack([np.interp(grid,arc,xy[:,k]) for k in (0,1)])
                xy-=xy.mean(0);scale=np.sqrt(np.mean(xy**2))
                if scale<=1e-8:continue
                xy/=scale;delta=np.diff(xy,axis=0);lengths=np.linalg.norm(delta,axis=1)
                delta=delta[lengths>1e-8];lengths=lengths[lengths>1e-8]
                if len(delta)<5:continue
                turns=np.arctan2(delta[:-1,0]*delta[1:,1]-delta[:-1,1]*delta[1:,0],(delta[:-1]*delta[1:]).sum(1))
                radius=np.linalg.norm(xy,axis=1)
                v=[np.linalg.norm(xy[-1]-xy[0])/lengths.sum(),lengths.std()/max(lengths.mean(),1e-8),np.median(np.abs(turns)),np.quantile(np.abs(turns),.9),radius.std()/max(radius.mean(),1e-8),abs(turns.sum())/max(np.abs(turns).sum(),1e-8)]
                if np.isfinite(v).all():blocks.append(v)
        if blocks:values=dict(zip(SHAPE_FEATURE_NAMES,np.mean(blocks,axis=0)))
        result.update({f'shape_{task}_{k}':float(v) for k,v in values.items()})
        quality.append(dict(task=task,point_count=len(ids),usable_segments=len(blocks),status='VALID' if blocks else 'INSUFFICIENT_USABLE_GEOMETRY'))
    return result,quality
