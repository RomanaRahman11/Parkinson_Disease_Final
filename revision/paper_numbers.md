# Numbers for the manuscript (FULL mode, 10 repeats, generated 2026-11-09 22:09)

- **T1 MRI**: BA 0.449 ± 0.058 (range 0.342–0.542); AUC 0.429 ± 0.063 (range 0.317–0.567); bootstrap AUC CI 0.363–0.767; permutation B=200, null BA 0.502, p_BA=0.308, p_AUC=0.294 → NOT beyond chance
- **Spiral drawing**: BA 0.692 ± 0.094 (range 0.567–0.858); AUC 0.742 ± 0.072 (range 0.650–0.838); bootstrap AUC CI 0.679–0.963; permutation B=200, null BA 0.504, p_BA=0.005, p_AUC=0.015 → beyond chance
- **Voice**: BA 0.606 ± 0.074 (range 0.528–0.722); AUC 0.685 ± 0.094 (range 0.500–0.806); bootstrap AUC CI 0.463–0.963; permutation B=200, null BA 0.518, p_BA=0.502, p_AUC=0.085 → NOT beyond chance

## Family selection
   modality                  family  times_selected
0       mri    MedicalNet_Global_LR               6
1       mri  MedicalNet_Regional_LR              10
2       mri                Voxel_LR               4
3       mri               Voxel_SVM              10
4    spiral        GradientBoosting               2
5    spiral      LogisticRegression               3
6    spiral                 RBF_SVM              20
7    spiral            RandomForest               4
8    spiral              SoftVoting               1
9     voice        GradientBoosting               1
10    voice      LogisticRegression              18
11    voice                 RBF_SVM               7
12    voice            RandomForest               4

## Holdout
                     set modality   n    AUC  AUC_HM_lo  AUC_HM_hi     BA  Brier  Brier_prevalence_reference  majority_baseline_accuracy  P_ge_observed_if_majority_rate
1  retrospective_holdout    voice   8  1.000      1.000      1.000  0.750  0.082                       0.188                       0.750                           0.367
3  retrospective_holdout   spiral   8  0.933      0.745      1.000  0.800  0.181                       0.234                       0.625                           0.370
5  retrospective_holdout      mri  11  0.536      0.169      0.903  0.536  0.261                       0.234                       0.636                           0.827

## Training vs outer
                   family  training_accuracy  accuracy  training_balanced_accuracy  balanced_accuracy modality
0        GradientBoosting              1.000     0.792                       1.000              0.694    voice
1      LogisticRegression              0.917     0.792                       0.889              0.694    voice
2                 RBF_SVM              1.000     0.833                       1.000              0.722    voice
3            RandomForest              1.000     0.792                       1.000              0.694    voice
4              SoftVoting              1.000     0.750                       1.000              0.611    voice
0        GradientBoosting              0.985     0.845                       0.988              0.823   spiral
1      LogisticRegression              0.781     0.621                       0.816              0.641   spiral
2                 RBF_SVM              0.906     0.685                       0.925              0.692   spiral
3            RandomForest              0.969     0.782                       0.975              0.754   spiral
4              SoftVoting              0.905     0.724                       0.924              0.728   spiral
0    MedicalNet_Global_LR              0.875     0.558                       0.875              0.548      mri
1  MedicalNet_Regional_LR              0.764     0.436                       0.769              0.417      mri
2                Voxel_LR              0.969     0.473                       0.975              0.417      mri
3               Voxel_SVM              0.534     0.558                       0.500              0.500      mri

## Spiral strata (dev)
label         control  PD
acq_stratum              
median_dt<=8       11   0
median_dt>8         1  20

## Spiral strata (all 40)
label   control  PD
row_0              
<=8 ms       14   0
>8 ms         1  25

## Spiral acquisition-controlled
                                      ba           auc       
                                    mean    std   mean    std
variant                                                      
acquisition_only                   0.958  0.060  0.978  0.048
shape_only                         0.643  0.149  0.645  0.155
shape_plus_acquisition             0.921  0.074  0.978  0.039
shape_residualised_on_acquisition  0.513  0.156  0.569  0.188

## Sensitivity
                                                       balanced_accuracy              roc_auc             
                                                                    mean    std count    mean    std count
analysis                                                                                                  
MRI_QC_pass_only                                                   0.348  0.086    10   0.335  0.134    10
voice_excluding_phon_R01_S21+phon_R01_S27+phon_R01_S35             0.614  0.098    10   0.645  0.104    10

## Voice leakage
                                                accuracy            BA           AUC        shared_identifiers       
                                                    mean    std   mean    std   mean    std               mean    std
scheme                      model                                                                                    
Recording-level 3-fold CV   Logistic regression    0.853  0.008  0.776  0.013  0.893  0.012               29.4  0.843
                            RBF-SVM                0.898  0.016  0.827  0.021  0.942  0.021               29.4  0.843
                            Random forest          0.898  0.011  0.825  0.016  0.950  0.010               29.4  0.843
Recording-level 80/20 split Logistic regression    0.826  0.055  0.755  0.100  0.901  0.054               22.8  1.687
                            RBF-SVM                0.910  0.042  0.874  0.062  0.963  0.033               22.8  1.687
                            Random forest          0.887  0.063  0.842  0.089  0.947  0.043               22.8  1.687
Subject-level 3-fold CV     Logistic regression    0.753  0.031  0.633  0.059  0.706  0.116                0.0  0.000
                            RBF-SVM                0.734  0.031  0.588  0.031  0.597  0.062                0.0  0.000
                            Random forest          0.762  0.035  0.612  0.053  0.752  0.044                0.0  0.000

## MRI QC
label  control  PD
flag              
False        6  16
True        10  11
Fisher p=0.215; Dice Mann-Whitney p=0.274; QC-pass dev counts {1: 10, 0: 4}