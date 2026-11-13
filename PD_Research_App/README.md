# Parkinson's research demo (independent models)

Research use only; not a medical device; scores are uncalibrated.

```
pip install -r requirements.txt
streamlit run app.py
```

- Voice: CSV with `subject_id` + 22 acoustic features (template included); recordings of one person are averaged.
- Spiral: raw semicolon TXT from the UCI tablet protocol, or one feature row.
- MRI: `normalized.npy` + `inference_metadata.json` produced by the notebook's preprocessing protocol.
