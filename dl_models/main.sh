#! /bin/bash
func_run(){
  time_step=$1
  data_file_name=$2
  static_features_path=$3
  # mor
  echo "==========================================================================="
  echo "$time_step => mor.."
  for(( num=0; num<6; num++))
  do
    echo "$time_step => mor => $num.."
    echo "==========================================================================="
    python main.py mor mor 1 --label_type $num --time_step $time_step --data_file_name $data_file_name
    #echo "==========================================================================="
    #python main.py mor mor 2 --label_type $num --n_features 414 --time_step $time_step --static_features_path $static_features_path
  done
  # los
  echo "==========================================================================="
  echo "$time_step => los.."
  echo "==========================================================================="
  python main.py los los 1 --learning_rate 0.005 --time_step $time_step --data_file_name $data_file_name
  echo "==========================================================================="
  python main.py los los 1 --learning_rate 0.005 --time_step $time_step --data_file_name $data_file_name --without_static
  echo "==========================================================================="
  python main.py los los 2 --learning_rate 0.005 --n_features 414 --time_step $time_step --static_features_path $static_features_path --data_file_name $data_file_name
  # icd9
  echo "==========================================================================="
  echo "$time_step => icd9.."
  for(( num=0; num<20; num++))
  do
    echo "$time_step => icd9 => $num.."
    echo "==========================================================================="
    python main.py icd9 icd9 1 --label_type $num --time_step $time_step --data_file_name $data_file_name
    echo "==========================================================================="
    python main.py icd9 icd9 1 --label_type $num --time_step $time_step --data_file_name $data_file_name --without_static
    echo "==========================================================================="
    python main.py icd9 icd9 2 --label_type $num --n_features 414 --time_step $time_step --static_features_path $static_features_path --data_file_name $data_file_name
  done
}
# 24 hours
echo "Running 24 hours data.."
data_file_name=/data3/Benchmarking_DL_MIMICIII/Data/admdata_99p/24hrs_raw/series/imputed-normed-ep_1_24.npz
static_features_path=/data3/Benchmarking_DL_MIMICIII/Data/admdata_99p/24hrs_raw/non_series/tsmean_24hrs.npz
time_step=24
func_run $time_step $data_file_name $static_features_path
echo "==========================================================================="
# 48 hours
echo "Running 48 hours data.."
data_file_name=/data3/Benchmarking_DL_MIMICIII/Data/admdata_99p/48hrs_raw/series/imputed-normed-ep_1_48.npz
static_features_path=/data3/Benchmarking_DL_MIMICIII/Data/admdata_99p/48hrs_raw/non_series/tsmean_48hrs.npz
time_step=48
func_run $time_step $data_file_name $static_features_path