# Deep learning models

[task name] : mor icd9 los

[model type] : 1 => MMDL 2 -> FFN

## FFN: Feedforward Network

```
python [path to the main program('main.py')] 
[name of dataset] [task name] 2 
--data_file_name [name of imputed data]
--label_type [label type] 
--static_features_path [path to static features] 
--n_features 414
```


## MMDL

### Feedforward Network + LSTM
```
python [path to the main program('main.py')]
[name of dataset] [task name] 1
--data_file_name [name of imputed data]
 --label_type 0
--time_step 48
--learning_rate
--nb_epoch 200
```

### LSTM: LSTM only

add `--without_static`

# Training commands

time_step: 24, 48

data_file_name: imputed-normed-ep_1_24.npz imputed-normed-ep_1_48.npz

static_features_path: tsmean_24hrs.npz tsmean_48hrs.npz

## Task 1: length of stay

### Model: MMDL

```
python main.py los los 1 --learning_rate 0.005 --time_step 24 --data_file_name imputed-normed-ep_1_24.npz
```

### Model: FFN

```
python main.py los los 2 --learning_rate 0.005 --time_step 24 --static_features_path tsmean_24hrs.npz
```

## Task 2: icd-9 group

label_type: 0, 1, 2, ..., 19 => ('icd_group1', ... , 'icd_group20')   
 
### Model: MMDL

```
python main.py icd9 icd9 1 --label_type 0 --time_step 24 --data_file_name imputed-normed-ep_1_24.npz
```

### Model: FFN

```
python main.py icd9 icd9 2 --label_type 0 --time_step 24 --static_features_path tsmean_24hrs.npz
```

## Task 3: mortality

label_type: 0, 1, 2, 3, 4 => ('mor_inhosp', mor24', 'mor48', 'mor72', 'mor30d', 'mor1y')

### Model: MMDL

```
python main.py mor mor 1 --label_type 0 --time_step 24 --data_file_name imputed-normed-ep_1_24.npz
```

### Model: FFN

```
python main.py mor mor 2 --label_type 0 --time_step 24 --static_features_path tsmean_24hrs.npz
```
