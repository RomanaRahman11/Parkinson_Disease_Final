import gc
fixed=sitk.ReadImage(str(TEMPLATE_PATH),sitk.sitkFloat32)
fixed_mask=sitk.ReadImage(str(TEMPLATE_MASK),sitk.sitkUInt8)
ref=sitk.GetArrayFromImage(fixed_mask)>0
PROTOCOL=dict(version='9.0',n4_iterations=[30,20,10],n4_estimation_shrink=[2,2,2],bias_application='native full-resolution grid before registration',
 registration='masked rigid Euler3D; mask-moment initialization',metric='MattesMutualInformation',metric_bins=40,
 sampling='RANDOM',sampling_fraction=.25,seed=SEED,optimizer='RegularStepGradientDescent',learning_rate=1.,
 min_step=.0001,iterations=250,relaxation_factor=.6,optimizer_scales='physical_shift',
 pyramid_shrink=[4,2,1],pyramid_smoothing_mm=[2,1,0],interpolator='linear',mask_interpolator='nearest',
 normalization=dict(percentiles=[1,99],scope='subject own aligned brain mask',clip=[0,1],background=0),
 array_order='SimpleITK GetArrayFromImage: z,y,x; physical reference grid recorded separately',
 reference_grid=dict(size=list(fixed.GetSize()),spacing=list(fixed.GetSpacing()),origin=list(fixed.GetOrigin()),direction=list(fixed.GetDirection())),
 slice_sampling=dict(fractions=[.25,.75],frozen_per_axis=FROZEN_SLICES_PER_VIEW,cnn_per_axis=CNN_SLICES_PER_VIEW,
 resize=[224,224],cnn_resize=[160,160],imagenet_mean=[.485,.456,.406],imagenet_std=[.229,.224,.225]),
 synthstrip_command='python mri_synthstrip -i INPUT -o BRAIN -m MASK --model WEIGHTS --threads 2',
 assets=ASSET_HASHES,software={p:__import__(p).__version__ for p in ['numpy','nibabel','SimpleITK']})
PROTOCOL['implementation_sha256']=sha256(OUT/'mri_preprocessing_cell.py')
PROTOCOL_HASH=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()
write_json(OUT/'tables/mri_preprocessing_protocol.json',PROTOCOL)

def preprocess_one(row,reference_mask_array=ref):
    deadline_check();folder=OUT/'mri_cache'/row.subject_id;folder.mkdir(exist_ok=True)
    marker=folder/'completed.json'
    if marker.exists():
        record=json.loads(marker.read_text())
        assert record['protocol_hash']==PROTOCOL_HASH and record['input_hash']==row.sha256,'Cache protocol/input mismatch; use a fresh output directory.'
        assert sha256(folder/'normalized.npy')==record['array_hash'],'Cache array corrupted'
        for filename,key in [('mask.nii.gz','native_mask_hash'),('registered_mask.nii.gz','registered_mask_hash'),('n4_corrected_native_brain.nii.gz','native_corrected_hash')]:
            assert sha256(folder/filename)==record[key],'Review image cache corrupted: '+filename
        return record
    brain=folder/'brain.nii.gz';mask=folder/'mask.nii.gz'
    # Unfinished cache is not trusted after a changed input/protocol.
    for stale in [brain,mask]:
        if stale.exists():stale.unlink()
    if not (brain.exists() and mask.exists()):
        cmd=[sys.executable,str(SCRIPT),'-i',row.path,'-o',str(brain),'-m',str(mask),'--model',str(WEIGHTS),'--threads','2']
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=1200)
        (folder/'synthstrip.log').write_text(result.stdout+'\n'+result.stderr)
        if result.returncode:raise RuntimeError(f'SynthStrip failed for {row.subject_id}; see log: {result.stderr[-500:]}')
    original=sitk.ReadImage(row.path,sitk.sitkFloat32);bm=sitk.ReadImage(str(mask),sitk.sitkUInt8)
    assert original.GetSize()==bm.GetSize()
    np.testing.assert_allclose(original.GetSpacing(),bm.GetSpacing(),atol=1e-4)
    np.testing.assert_allclose(original.GetDirection(),bm.GetDirection(),atol=1e-4)
    original_array=sitk.GetArrayFromImage(original)
    assert np.isfinite(original_array).all() and np.any(original_array!=0),'Non-finite or empty MRI voxels'
    del original_array
    np.testing.assert_allclose(original.GetOrigin(),bm.GetOrigin(),atol=1e-4)
    # বাংলা: ছোট image দিয়ে bias field estimate; corrected image full-resolution-এ থাকবে।
    positive_mask=sitk.Cast((bm>0)&(original>0),sitk.sitkUInt8)
    small=sitk.Shrink(original,[2,2,2]);small_mask=sitk.Shrink(positive_mask,[2,2,2])
    n4=sitk.N4BiasFieldCorrectionImageFilter();n4.SetMaximumNumberOfIterations([30,20,10])
    n4.Execute(small,small_mask)
    log_bias=n4.GetLogBiasFieldAsImage(original)
    corrected=sitk.Cast(original/sitk.Exp(log_bias),sitk.sitkFloat32)
    ms=sitk.Cast(bm>0,sitk.sitkUInt8)
    moving=sitk.Mask(corrected,ms)
    assert moving.GetSize()==original.GetSize(),'Native resolution must be preserved.'
    sitk.WriteImage(moving,str(folder/'n4_corrected_native_brain.nii.gz'))
    native_mask_array=sitk.GetArrayFromImage(ms)>0
    native_brain_volume_ml=float(native_mask_array.sum()*np.prod(original.GetSpacing())/1000.)
    native_mask_fraction=float(native_mask_array.mean())
    from scipy.ndimage import label as component_label
    components,count=component_label(native_mask_array)
    sizes=np.bincount(components.ravel())[1:]
    largest_component_fraction=float(sizes.max()/sizes.sum()) if len(sizes) else 0.
    del components,native_mask_array
    initial=sitk.CenteredTransformInitializer(sitk.Cast(fixed_mask,sitk.sitkFloat32),sitk.Cast(ms,sitk.sitkFloat32),sitk.Euler3DTransform(),sitk.CenteredTransformInitializerFilter.MOMENTS)
    reg=sitk.ImageRegistrationMethod();reg.SetMetricAsMattesMutualInformation(40)
    reg.SetMetricFixedMask(fixed_mask);reg.SetMetricMovingMask(ms)
    reg.SetMetricSamplingStrategy(reg.RANDOM);reg.SetMetricSamplingPercentage(.25,SEED)
    reg.SetInterpolator(sitk.sitkLinear);reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.,minStep=.0001,numberOfIterations=250,relaxationFactor=.6)
    reg.SetOptimizerScalesFromPhysicalShift();reg.SetShrinkFactorsPerLevel([4,2,1]);reg.SetSmoothingSigmasPerLevel([2,1,0]);reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial,inPlace=False);transform=reg.Execute(fixed,moving)
    aligned=sitk.Resample(moving,fixed,transform,sitk.sitkLinear,0.,sitk.sitkFloat32)
    aligned_mask=sitk.Resample(ms,fixed,transform,sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
    optimizer_metric=float(reg.GetMetricValue())
    if not np.isfinite(optimizer_metric):raise RuntimeError('Non-finite registration metric.')
    a=sitk.GetArrayFromImage(aligned).astype('float32');m=sitk.GetArrayFromImage(aligned_mask)>0
    assert np.isfinite(a).all() and m.sum()>100
    # বাংলা: intensity normalization প্রতি subject-এর নিজের brain mask থেকে; cohort statistics নয়।
    lo,hi=np.percentile(a[m],[1,99]);assert hi>lo
    normalized=np.clip((a-lo)/(hi-lo),0,1);normalized[~m]=0
    np.save(folder/'normalized.npy',normalized.astype('float32'))
    sitk.WriteImage(aligned,str(folder/'registered_brain.nii.gz'));sitk.WriteImage(aligned_mask,str(folder/'registered_mask.nii.gz'))
    sitk.WriteTransform(transform,str(folder/'rigid_transform.h5'))
    intersection=(reference_mask_array&m).sum();dice=2*intersection/(reference_mask_array.sum()+m.sum());coverage=intersection/reference_mask_array.sum()
    active=np.where(reference_mask_array.sum(axis=(1,2))>0)[0];edge=active[:max(1,len(active)//5)]
    inferior=float((reference_mask_array[edge]&m[edge]).sum()/reference_mask_array[edge].sum())
    fig,axes=plt.subplots(3,2,figsize=(8,10),constrained_layout=True)
    for axidx,axis in enumerate([0,1,2]):
        i=a.shape[axis]//2
        axes[axidx,0].imshow(np.rot90(np.take(normalized,i,axis=axis)),cmap='gray',vmin=0,vmax=1)
        axes[axidx,1].imshow(np.rot90(np.take(normalized,i,axis=axis)),cmap='gray',vmin=0,vmax=1)
        axes[axidx,1].contour(np.rot90(np.take(reference_mask_array,i,axis=axis)),levels=[.5],colors=['red'],linewidths=.5)
        for ax in axes[axidx]:ax.axis('off')
    fig.suptitle(f'{row.subject_id} | anatomical QC pending | Dice={dice:.3f}')
    fig.savefig(folder/'qc.png',dpi=150);plt.close(fig)
    record=dict(subject_id=row.subject_id,protocol_hash=PROTOCOL_HASH,input_hash=row.sha256,
       array_hash=sha256(folder/'normalized.npy'),dice=float(dice),coverage=float(coverage),inferior_coverage=inferior,
       native_mask_hash=sha256(mask),registered_mask_hash=sha256(folder/'registered_mask.nii.gz'),native_corrected_hash=sha256(folder/'n4_corrected_native_brain.nii.gz'),
       native_brain_volume_ml=native_brain_volume_ml,native_mask_fraction=native_mask_fraction,largest_component_fraction=largest_component_fraction,registration_metric=optimizer_metric,
       priority_review=bool(dice<.82 or coverage<.75 or inferior<.5 or largest_component_fraction<.95),stop_condition=reg.GetOptimizerStopConditionDescription(),visual_qc='PENDING')
    write_json(marker,record);gc.collect();return record

complete('V9-09')
