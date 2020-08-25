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

---

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

---

# Experiment Result

## Task 1: length of stay

|MSE|24hrs|48hrs|	
|---|---|---|
|FFN|	63713| 61390 |
|RNN|	64846| 61390 |
|MMDL|	63713| 61390 |

## Task 2: icd-9 group

|ICD9 GROUP| PROPORTION |FFN AUROC|FFN AUPRC|RNN AUROC|RNN AUPRC|MMDL AUROC|MMDL AUPRC|
|---|---|---|---|---|---|---|---|
|1|0.253|0.788|0.590|0.788|0.594|0.790|0.590|
|2|0.172|0.854|0.748|0.772|0.510|0.840|0.730|
|3|0.684|0.731|0.842|0.736|0.851|0.734|0.849|
|4|0.366|0.767|0.667|0.772|0.672|0.771|0.667|
|5|0.317|0.676|0.511|0.683|0.514|0.682|0.513|
|6|0.289|0.674|0.463|0.701|0.507|0.697|0.502|
|7|0.829|0.853|0.961|0.817|0.950|0.818|0.947|
|8|0.481|0.748|0.745|0.778|0.782|0.776|0.779|
|9|0.389|0.722|0.663|0.730|0.678|0.728|0.672|
|10|0.394|0.805|0.763|0.814|0.775|0.816|0.779|
|11|0.004|0.808|0.190|0.709|0.083|0.728|0.091|
|12|0.101|0.678|0.203|0.640|0.187|0.646|0.189|
|13|0.188|0.636|0.277|0.639|0.293|0.641|0.291|
|14|0.036|0.678|0.112|0.593|0.053|0.601|0.067|
|15|0.318|0.675|0.500|0.692|0.519|0.630|0.450|
|16|0.084|0.610|0.135|0.574|0.123|0.575|0.120|
|17|0.030|0.678|0.066|0.563|0.050|0.599|0.057|
|18|0.449|0.688|0.657|0.686|0.653|0.687|0.655|
|19|0.467|0.698|0.661|0.708|0.676|0.709|0.678|
|20|0.333|0.684|0.560|0.703|0.579|0.707|0.583|


## Task 3: mortality

### Mortality Info

|In-hospital|2-day|3-day|30-day|1-year|TOTAL|
| --- | --- | --- | --- | --- | --- |
|0.099|	0.016|	0.027|	0.138|	0.241|	35623|


### Data within 24 hours

| label | AUROC | AUPRC |
| ----  | ----  | ----  |
|inhsp	|0.865 	|0.567 |
|2days	|0.691 	|0.116 |
|3days	|0.739 	|0.233 |
|30days	|0.870 	|0.602 |
|1year	|0.848 	|0.664 |


### Data within 48 hours

| label | AUROC | AUPRC |
| ----  | ----  | ----  |
|inhsp	|0.854 	|0.559 |
|3days	|0.690 	|0.162 |
|30days	|0.850 	|0.590 |
|1year	|0.842 	|0.653 |
