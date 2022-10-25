#!/bin/bash

export PATH="/home/poker/miniconda3/bin:$PATH"
#export time=`date -u "+%Y%m%d%H%M" -d "6 min ago"`

cd /home/poker/goes17/conusc

#  /weather/data/goes16/"+prod_id+"/"+band+"/latest.nc
cp /weather/data/goes16/TIRW/10/latest.nc /dev/shm/latest_TIRW_10.nc
cmp /weather/data/goes16/TIRW/10/latest.nc /dev/shm/latest_TIRW_10.nc > /dev/null
CONDITION=$?
#echo $CONDITION

while :; do

  until [ $CONDITION -eq 1 ] ; do
#     echo same
     sleep 5
     cmp /weather/data/goes16/TIRW/10/latest.nc /dev/shm/latest_TIRW_10.nc > /dev/null
     CONDITION=$?
  done

#  echo different
  cp /weather/data/goes16/TIRW/10/latest.nc /dev/shm/latest_TIRW_10.nc
  sleep 52
  /home/poker/miniconda3/bin/python goes17_conusc_wvl_fixeddisk.py /dev/shm/latest_TIRW_10.nc
  cmp /weather/data/goes16/TIRW/10/latest.nc /dev/shm/latest_TIRW_10.nc > /dev/null
  CONDITION=$?
#  echo repeat

done


python goes17_conusc_wvl.py $time


