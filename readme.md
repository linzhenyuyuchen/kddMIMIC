# Table pre-processing

## Create event tables for useful itemids

Events tables : "inputevents_cv", "inputevents_mv", "chartevents", "labevents", "outputevents" 

Run `./extract_tables.py` in which we filter useful itemids and drugs by `variable_map` and change their units to be consistent,
 and drop records whose units we cannot change by rules for further extration.

## Break up by subject

Run `./extract_subjects.py` in which we extract the features from first icu admission for 38597 distinct adult subjects.

For `icustays`, add age, and given-hours mortality, only keep age > 15

For `prescriptions`, `diagnoses` and `procedures`, filter them on `icustays` ,and count the number of occurrences of ICD-9 code 
which we can use to keep only icd9 code by the number of occurrences above threshold.


---

Break up basic tables above and events tables by subject respectively:

Filter records whose value is nan or empty string.


## Validate events

Run `./validate_events.py`

1. drop all events whose HADM_ID is empty or not listed in `stays.csv`

2. recover ICUSTAY_ID if it is empty in `stays.csv`

3. drop all events whose ICUSTAY_ID is not the same as that of stays.csv for the same HADM_ID

4. filter events with nan VALUE

5. save at "X_verified.csv"

## Extract episode from subjects

Run `./extract_episodes_from_subjects.py` in which we extract raw timeseries data in `episodeN_timeseries_processed.csv` 
, processed timeseries data in `episodeN_timeseries_raw.csv` , and static episode including basic info and diagnoses 
icd9 codes group in `episodeN.csv`.

Besides, there is `episodes_index` in output path which save 38466 SUBJECT_ID to its episode path.

`stays.csv` + `diagnoses.csv` + `procedures.csv` >> episode.csv

`prescriptions`, `inputevents_cv`, `inputevents_mv`, `labevents`,`outputevents`, `chartevents` >> 
episode_timeseries.csv

## Create dataset for experiments

Run `./create_dataset.py` in which we creat dataset from 38462 subjects for experiments.

 EOF

