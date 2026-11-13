"""Parkinson's research demo — independent voice / spiral / MRI models (research use only)."""
from pathlib import Path
import io, json, tempfile
import numpy as np, pandas as pd, joblib, streamlit as st
from pd_four_model_core import model_score, voxel_vector, spatial_vector, usable_input_rows
from pd_preprocessing_core import load_trajectory
from pd_spiral_features import shape_features
from pd_input_validation import validate_mri_metadata, validate_feature_frame

ROOT = Path(__file__).parent
cfg = json.loads((ROOT / "input_config.json").read_text())
schema = json.loads((ROOT / "feature_schema.json").read_text())
status = json.loads((ROOT / "workflow_status.json").read_text())

st.set_page_config(page_title="Parkinson Research Demo", page_icon="🧠", layout="centered")
st.title("Parkinson's disease — research demo")
st.error("RESEARCH ONLY — not a medical device. Scores are uncalibrated. In the accompanying study, voice was not "
         "distinguishable from chance, spiral discrimination was explained by acquisition timing, and MRI showed no signal.")
with st.expander("Study status flags"):
    st.json(status)

name = st.selectbox("Modality", ["voice", "spiral", "mri"],
                    format_func={"voice": "Voice (acoustic-feature CSV)", "spiral": "Spiral drawing (raw TXT or feature CSV)",
                                 "mri": "T1 MRI (normalised NPY + metadata JSON)"}.get)
template = ROOT / f"{name}_feature_template.csv"
if name != "mri" and template.exists():
    st.download_button("Download input template", template.read_bytes(), file_name=template.name)
upload = st.file_uploader("Upload input", type=["npy"] if name == "mri" else (["csv", "txt"] if name == "spiral" else ["csv"]))
meta = st.file_uploader("Matching inference_metadata.json", type=["json"]) if name == "mri" else None

@st.cache_resource
def bundle_for(n):
    return joblib.load(ROOT / f"{n}_selected_bundle.joblib")

@st.cache_resource
def medicalnet():
    from pd_medicalnet import load_encoder
    enc, _ = load_encoder(ROOT / "medicalnet_resnet.py", ROOT / "resnet_18_23dataset.pth", "cpu")
    return enc

if upload is not None and st.button("Run research prediction", type="primary"):
    try:
        bundle = bundle_for(name); model = bundle["model"]
        st.caption(f"Locked model: **{bundle['family']}**, threshold {float(bundle['threshold']):.2f}")
        if name == "mri":
            if meta is None: raise ValueError("MRI requires the matching inference_metadata.json.")
            validate_mri_metadata(json.loads(meta.getvalue()), upload.getvalue(), cfg["protocol_hash"])
            a = np.load(io.BytesIO(upload.getvalue()), allow_pickle=False)
            if list(a.shape) != cfg["mri_shape"] or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
                raise ValueError("Expected the trained MRI grid with finite normalised values in [0, 1].")
            fam = bundle["family"]
            if fam.startswith("MedicalNet_"):
                from pd_medicalnet import extract
                g, r = extract(a, cfg["reference_grid"], medicalnet(), "cpu")
                p = float(model_score(model, (g if "Global" in fam else r)[None, :])[0])
            elif fam.startswith("Spatial_"): p = float(model_score(model, spatial_vector(a)[None, :])[0])
            else: p = float(model_score(model, voxel_vector(a)[None, :])[0])
        else:
            if name == "spiral" and upload.name.lower().endswith(".txt"):
                with tempfile.TemporaryDirectory() as d:
                    path = Path(d) / "trajectory.txt"; path.write_bytes(upload.getvalue())
                    feats, quality = shape_features(load_trajectory(path), cfg["spiral_resampling_points"])
                frame = pd.DataFrame([feats]); st.write("Task quality", quality)
            else:
                frame = pd.read_csv(upload)
            missing = set(schema[name]) - set(frame)
            if missing: raise ValueError(f"Missing features: {sorted(missing)}")
            if name == "voice" and ("subject_id" not in frame or frame.subject_id.nunique() != 1):
                raise ValueError("Voice CSV needs one subject_id shared by all of that person's recordings.")
            frame, values = validate_feature_frame(frame, schema[name], name)
            if name == "spiral":
                if len(values) != 1: raise ValueError("Provide exactly one subject feature row.")
                if not usable_input_rows(bundle, values).all(): raise ValueError("ABSTAIN: selected task has no usable geometry.")
                p = float(model_score(model, values)[0])
            else:
                p = float(model_score(model, values).mean())     # mean over the person's recordings
        thr = float(bundle["threshold"])
        c1, c2 = st.columns(2)
        c1.metric("Uncalibrated PD score", f"{p:.3f}")
        c2.metric("Research class", "PD" if p >= thr else "Control")
        st.caption("Not a clinical probability. No combined multimodal score is produced.")
    except Exception as e:
        st.error(str(e))
