
# Data Frame

## Raw Data

`ADM_FEATURES_24hrs.npy` : `list [ N_samples, array (age, AIDS,hematologic malignancy, metastatic cancer, admission type) ]`

---

`ADM_LABELS_24hrs.npy` : `list [ N_samples, array ('mor_inhosp', mor24', 'mor48', 'mor72', 'mor30d', 'mor1y') ]`

**Mortality label here can be used for the paper.**

---

`AGE_LOS_MORTALITY_24hrs.npy` : `list [ N_samples, list [ as below] ]`

```
Here we collect all non-temporal features only related to the admissions:
1. admission id
2. subject id(for finding the patient of one admission)
3. age(at admittime, unit is day)
4. length of stay(unit is minute)
5. in-hospital mortality label
6. labelGurantee label
7. 1-day mortality(from admittime)
8. 2-day mortality(from admittime)
9. 3-day mortality(from admittime)
10. 30-day mortality(from dischtime)
11. 1-year mortality(from dischtime)
12. curr_service
13. admission_type
14. admission_location
15. insurance
16. language
17. religion
18. marital_status
19. ethnicity
```
**Mortality label here is not used, We leave them here only for compatibility.**

---

`ICD9-24hrs.npy` : `list [ N_samples, list [ as below] ]`

```
[
    [admission id, icd-9, icd-9-standard, group]
    ...
    ...
]
```

---

`DB_merged_24hrs.npy` : `list [ N_samples, list [ as below] ]`

```
[
    [feature 1, …, feature n(140), seconds of (current time-icu intime), admission_id]
    ...
    ...
]
```


## Non-series Data

`tsmean_24hrs.npz` : ["hrs_mean_array"]
 
 `list [ N_samples, array (min1 - min136 max1 - max136 mean1 - mean136 urinary_output_sum 
age, AIDS,hematologic malignancy, metastatic cancer, admission type) ]`

---


`tsmean_24hrs.npz` : ["hrs_mean_labels"]
 
 `list [ N_samples, array ('mor_inhosp', mor24', 'mor48', 'mor72', 'mor30d', 'mor1y') ]`
 
 
 ## Series Data
 
 
`normed-ep-stats.npz`

> this file including various statistical data

```
class_icd9_list
class_icd9_counts
keep_val_idx_list
ep_tdata_mean
ep_tdata_std
n_class_icd9
N
val_mr
idx_x
age_days
```

---

`normed-ep.npz`

```
X_t : list [ N_samples, array(n_times, 136) ]
X_t_mask
T_t: list [n_times, int(second)]
deltaT_t
y_icd9: list [ N_samples, array(20) ]
y_mor: list [n_times, int()]
adm_features_all
adm_labels_all
y_los: list [n_times, int()]
```

---

`imputed-normed-ep_1_24.npz`

```
ep_tdata : list [ N_samples, array(24Hours, 136) ]
adm_features_all: list [ N_samples, array(5) ]
y_icd9: list [ N_samples, array(20) ]
adm_labels_all: list [ N_samples, array(5) ]
y_los: list [N_samples, int()]
```



