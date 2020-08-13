# deep learning models

## FFN: Feedforward Network

```
python [path to the main program('main.py')] 
[name of dataset] [task name] 2 
--data_file_name [name of imputed data]
--label_type [label type] 
--static_features_path [path to static features, ‘input.csv’] 
--nb_epoch 200
```

[name of dataset] : mimic3_99p_raw_24h mimic3_99p_raw_48h

[task name] : mor icd9 los

[label type] : 0 1 2 3 4 5 => ('mor_inhosp', mor24', 'mor48', 'mor72', 'mor30d', 'mor1y')

```
python main.py mimic3_99p_raw_24h mor 2 
--data_file_name imputed-normed-ep_1_24.npz
--label_type 0
--static_features_path ./Data/admdata_99p/24hrs_raw/non_series/input.csv 
--nb_epoch 200
```

## LSTM: LSTM only

```
python [path to the main program('main.py')]
[name of dataset] [task name] 1
--data_file_name [name of imputed data]
--label_type 0
--without_static
--time_step 48
--learning_rate [0.001 for mortality and icd9 prediction and 0.005 for length of stay prediction]
--nb_epoch 200
```

```
python main.py
mimic3_99p_raw_24h mor 1 
--data_file_name imputed-normed-ep_1_24.npz
--label_type 0 
--without_static
--time_step 24
--learning_rate 0.001
--nb_epoch 200
```

## MMDL: Feedforward Network + LSTM

```
python [path to the main program('main.py')]
[name of dataset] [task name] 1
--data_file_name [name of imputed data]
 --label_type 0
--time_step 48
--learning_rate [0.001 for mortality and icd9 prediction and 0.005 for length of stay prediction]
--nb_epoch 200
```

```
python main.py 
mimic3_99p_raw_24h mor 1 
--data_file_name imputed-normed-ep_1_24.npz
--label_type 0
--time_step 24
--learning_rate 0.001
--nb_epoch 200
```
