"""Input/provenance validation, independent of torch and Streamlit."""
import hashlib
import numpy as np
import pandas as pd

def validate_qc_review(review,current):
    fields=['subject_id','input_hash','array_hash','protocol_hash','native_mask_hash','registered_mask_hash','native_corrected_hash']
    needed=set(fields+['decision','reviewer','evidence'])
    if not needed<=set(review) or not review.subject_id.is_unique:raise ValueError('Invalid QC review schema.')
    if set(review.subject_id)!=set(current.subject_id):raise ValueError('Review must cover every current subject.')
    joined=current[fields].merge(review,on='subject_id',validate='one_to_one',suffixes=('_current','_review'))
    for key in fields[1:]:
        if not joined[key+'_current'].eq(joined[key+'_review']).all():raise ValueError('Stale QC review: '+key+' does not match current preprocessing.')
    d=review.decision.astype(str).str.strip().str.lower()
    if not d.isin(['accept','reject']).all():raise ValueError('Each QC decision must be accept/reject.')
    for key in ['reviewer','evidence']:
        if review[key].isna().any() or review[key].astype(str).str.strip().eq('').any():raise ValueError('Nonempty QC '+key+' required.')
    if d.eq('reject').any():raise ValueError('Rejected scan: correct preprocessing and obtain a revised review; no automatic deletion.')
    return True

def validate_mri_metadata(metadata,payload,expected_protocol_hash):
    if metadata.get('protocol_hash')!=expected_protocol_hash:raise ValueError('MRI preprocessing protocol mismatch.')
    if metadata.get('array_hash')!=hashlib.sha256(payload).hexdigest():raise ValueError('MRI NPY bytes do not match metadata.')
    if metadata.get('array_order')!='z,y,x':raise ValueError('Expected trained z,y,x array ordering.')
    return True

def validate_feature_frame(frame,features,name):
    if frame.empty or frame.columns.duplicated().any():raise ValueError('Nonempty input with unique columns required.')
    if not set(features)<=set(frame):raise ValueError('Missing required feature columns.')
    if name=='voice':
        if 'subject_id' not in frame or frame.subject_id.isna().any() or frame.subject_id.astype(str).str.strip().eq('').any() or frame.subject_id.nunique()!=1:raise ValueError('Exactly one nonempty voice subject_id is required.')
        frame=frame.drop_duplicates(subset=['subject_id']+list(features))
    elif len(frame)!=1:raise ValueError('One extracted Spiral subject row is required.')
    values=frame[list(features)].apply(pd.to_numeric,errors='raise').to_numpy(float)
    if np.isinf(values).any():raise ValueError('Infinite features are invalid.')
    if name=='voice' and not np.isfinite(values).all():raise ValueError('Voice features must be finite.')
    if np.isnan(values).all():raise ValueError('ABSTAIN: no usable input features.')
    return frame,values
