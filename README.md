# Subject-Level Validation of Independent Voice, Spiral-Drawing and T1-Weighted MRI Classifiers for Parkinson's Disease

**An audit of small public cohorts** · Romana Rahman and Nahid Islam Nayem · Khwaja Yunus Ali University

> **Research use only.** Nothing in this repository is a medical device or a diagnostic tool.

**Project page:** https://romanarahman11.github.io/Parkinson_Disease_Final/

## Summary

Three public cohorts were audited separately: UCI voice (195 recordings, 32 identifiers),
UCI spiral drawing (40 files) and NEUROCON T1-weighted MRI (43 scans). With a fixed
logistic-regression primary analysis over ten subject-grouped partitions and permutation
nulls on the same partitions, balanced accuracy was 0.63 for voice (p = 0.12), 0.64 for spiral
(p = 0.039, not robust to multiplicity and explained by acquisition timing) and 0.46 for MRI
(two-sided p = 0.66). Splitting voice by recording instead of by person inflated BA by 0.14–0.24.

## Repository layout

| Folder / file | Contents |
|---|---|
| `notebook*.ipynb` | Executed analysis notebook |
| `PD_Research_App/` | Streamlit research demo, locked model bundles, preprocessing and validation code |
| `tables/` | Per-fold scores, inner searches, out-of-fold predictions, QC, splits and audits |
| `revision/` | Repeated nested CV, full-pipeline permutations, sensitivity and acquisition analyses |
| `figures/` | Model comparisons, learning curves, out-of-fold and hold-out plots |
| `index.html`, `assets/` | Project page served by GitHub Pages |

## Run the demo locally

```
cd PD_Research_App
pip install -r requirements.txt
streamlit run app.py
```

## Data and licences

UCI Parkinson's voice and spiral-drawing datasets and the NEUROCON MRI cohort are public and
are not redistributed here. MedicalNet weights (Tencent, MIT licence) are downloaded at run time;
see `PD_Research_App/README.md`.
