#! /bin/bash

func_run(){

  time_step=$1
  data_file_name=$2
  static_features_path=$3

  # los
  echo "$time_step => los.."
  python main.py los los 1 --learning_rate 0.005 --time_step $time_step --data_file_name $data_file_name
  python main.py los los 2 --learning_rate 0.005 --time_step $time_step --static_features_path static_features_path

  # icd9
  echo "$time_step => icd9.."
  for(( num=0; num<20; num++))
  do
    echo "$time_step => icd9 => $num.."
    python main.py icd9 icd9 1 --label_type $num --time_step $time_step --data_file_name $data_file_name
    python main.py icd9 icd9 2 --label_type $num --time_step $time_step --static_features_path static_features_path
  done

  # mor
  echo "$time_step => mor.."
  for(( num=0; num<5; num++))
  do
    echo "$time_step => mor => $num.."
    python main.py mor mor 1 --label_type $num --time_step $time_step --data_file_name $data_file_name
    python main.py mor mor 2 --label_type $num --time_step $time_step --static_features_path static_features_path
  done
}

# 24 hours
echo "Running 24 hours data.."
data_file_name=imputed-normed-ep_1_24.npz
static_features_path=tsmean_24hrs.npz
time_step=24
func_run $time_step $data_file_name $static_features_path

echo "========================"

# 48 hours
echo "Running 48 hours data.."
data_file_name=imputed-normed-ep_1_48.npz
static_features_path=tsmean_48hrs.npz
time_step=48
func_run $time_step $data_file_name $static_features_path
