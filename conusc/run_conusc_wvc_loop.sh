#!/bin/bash

export PATH="/home/poker/miniconda3/bin:$PATH"
#export time=`date -u "+%Y%m%d%H%M" -d "6 min ago"`

cd /home/poker/goes17/conusc

#  /weather/data/goes16/"+prod_id+"/"+band+"/latest.nc
cp /weather/data/goes16/TIRW/09/latest.nc /dev/shm/latest_TIRW_09.nc
cmp /weather/data/goes16/TIRW/09/latest.nc /dev/shm/latest_TIRW_09.nc > /dev/null
CONDITION=$?
#echo $CONDITION

while :; do

  until [ $CONDITION -eq 1 ] ; do
#     echo same
     sleep 5
     cmp /weather/data/goes16/TIRW/09/latest.nc /dev/shm/latest_TIRW_09.nc > /dev/null
     CONDITION=$?
  done

#  echo different
  cp /weather/data/goes16/TIRW/09/latest.nc /dev/shm/latest_TIRW_09.nc
  sleep 42
  /home/poker/miniconda3/bin/python goes17_conusc_wvc_fixeddisk.py /dev/shm/latest_TIRW_09.nc
  cmp /weather/data/goes16/TIRW/09/latest.nc /dev/shm/latest_TIRW_09.nc > /dev/null
  CONDITION=$?
#  echo repeat

done


python goes17_conusc_wvc.py $time


