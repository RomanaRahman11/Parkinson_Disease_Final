"""Frozen MedicalNet features: exploratory transfer, not a Parkinson diagnostic.
MedicalNet source/weights attribution: Tencent/MedicalNet, MIT; Chen et al. Med3D (2019).
"""
from pathlib import Path
import hashlib,json,importlib.util
import numpy as np

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
    return h.hexdigest()

def canonical_xyz(volume,grid):
    import nibabel as nib
    a=np.asarray(volume,np.float32)
    if a.ndim!=3 or not np.isfinite(a).all() or a.min()<0 or a.max()>1:raise ValueError('Expected finite [0,1] normalized MRI.')
    if tuple(a.shape[::-1])!=tuple(grid['size']):raise ValueError('Reference grid size mismatch.')
    affine=np.eye(4);affine[:3,:3]=np.asarray(grid['direction']).reshape(3,3)@np.diag(grid['spacing']);affine[:3,3]=grid['origin']
    affine=np.diag([-1,-1,1,1])@affine  # ITK physical LPS -> NIfTI RAS
    transform=nib.orientations.ornt_transform(nib.orientations.io_orientation(affine),nib.orientations.axcodes2ornt(('R','A','S')))
    return nib.orientations.apply_orientation(a.transpose(2,1,0),transform).copy()

def medical_input(volume,grid,shape=(96,112,96)):
    import torch
    from torch.nn import functional as F
    xyz=canonical_xyz(volume,grid);mask=xyz>0
    if mask.sum()<100:raise ValueError('Insufficient MRI foreground.')
    foreground=xyz[mask];sd=float(foreground.std())
    if sd<1e-6:raise ValueError('Degenerate MRI contrast.')
    normalized=np.zeros_like(xyz);normalized[mask]=np.clip((foreground-foreground.mean())/sd,-5,5)
    tensor=torch.from_numpy(normalized)[None,None]
    return F.interpolate(tensor,size=shape,mode='trilinear',align_corners=False)

def load_encoder(source,weights,device='cpu'):
    import torch
    spec=importlib.util.spec_from_file_location('medicalnet_resnet_source',str(source));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    encoder=module.resnet18(sample_input_D=96,sample_input_H=112,sample_input_W=96,num_seg_classes=2,shortcut_type='A',no_cuda=(device=='cpu'))
    encoder.conv_seg=torch.nn.Identity()
    checkpoint=torch.load(weights,map_location='cpu',weights_only=True)
    raw=checkpoint.get('state_dict',checkpoint)
    if not isinstance(raw,dict):raise ValueError('Checkpoint has no tensor state dictionary.')
    state={k.removeprefix('module.'):v for k,v in raw.items() if not k.removeprefix('module.').startswith('conv_seg.')}
    expected=encoder.state_dict()
    unexpected=set(state)-set(expected);missing=set(expected)-set(state)
    critical_missing=[k for k in missing if not k.endswith('num_batches_tracked')]
    mismatched=[k for k in state if k in expected and tuple(state[k].shape)!=tuple(expected[k].shape)]
    if unexpected or critical_missing or mismatched:raise RuntimeError(f'Pretrained backbone mismatch: unexpected={unexpected}, missing={critical_missing}, shape={mismatched}')
    result=encoder.load_state_dict(state,strict=False)
    if any(not k.endswith('num_batches_tracked') for k in result.missing_keys) or result.unexpected_keys:raise RuntimeError('Incomplete backbone load.')
    for parameter in encoder.parameters():parameter.requires_grad=False
    encoder.to(device).eval()
    return encoder,dict(loaded_tensors=len(state),missing_only_bn_counters=sorted(missing),source_hash=digest(source),weights_hash=digest(weights))

def extract(volume,grid,encoder,device='cpu'):
    import torch
    from torch.nn import functional as F
    x=medical_input(volume,grid).to(device)
    with torch.inference_mode():
        z=encoder(x)
        if z.ndim!=5 or z.shape[1]!=512 or not torch.isfinite(z).all():raise RuntimeError('Unexpected pretrained backbone output.')
        global_features=torch.cat([F.adaptive_avg_pool3d(z,1).flatten(1),F.adaptive_max_pool3d(z,1).flatten(1)],1)
        regional_features=F.adaptive_avg_pool3d(z,(2,2,2)).flatten(1)
    return global_features.cpu().numpy()[0].astype('float32'),regional_features.cpu().numpy()[0].astype('float32')
