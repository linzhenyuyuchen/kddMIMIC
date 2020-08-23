#! /bin/bash

# 24 hours
echo "Running 24 hours data.."
data_file_name=/data3/Benchmarking_DL_MIMICIII/Data/admdata_99p/24hrs_raw/series/imputed-normed-ep_1_24.npz
static_features_path=/data3/Benchmarking_DL_MIMICIII/Data/admdata_99p/24hrs_raw/non_series/tsmean_24hrs.npz
time_step=24

for num in 0 2 3 4 5
do
  echo "$time_step => mor => $num.."
  echo "==========================================================================="
  python main.py mor mor 1 --label_type $num --time_step $time_step --data_file_name $data_file_name
done

echo "==========================================================================="

# 48 hours
echo "Running 48 hours data.."
data_file_name=/data3/Benchmarking_DL_MIMICIII/Data/admdata_99p/48hrs_raw/series/imputed-normed-ep_1_48.npz
static_features_path=/data3/Benchmarking_DL_MIMICIII/Data/admdata_99p/48hrs_raw/non_series/tsmean_48hrs.npz
time_step=48

for num in 0 3 4 5
do
  echo "$time_step => mor => $num.."
  echo "==========================================================================="
  python main.py mor mor 1 --label_type $num --time_step $time_step --data_file_name $data_file_name
done
