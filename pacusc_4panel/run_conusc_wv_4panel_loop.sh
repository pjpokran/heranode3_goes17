#!/bin/bash

export PATH="/home/poker/miniconda3/bin:$PATH"
#export time=`date -u "+%Y%m%d%H%M" -d "6 min ago"`
#export time=`ls -1 /weather/data/goes16/TIRE/09/*PAA.nc | awk '{$1 = substr($1,30,12)} 1' | sort -u | tail -2 | head -1`
#
#echo $time
#
#sleep 92

cd /home/poker/goes16/conusc_4panel

#  /weather/data/goes16/"+prod_id+"/"+band+"/latest.nc
cp /weather/data/goes16/TIRE/08/latest.nc /dev/shm/latest_TIRE_4panelwvir.nc
cmp /weather/data/goes16/TIRE/08/latest.nc /dev/shm/latest_TIRE_4panelwvir.nc > /dev/null
CONDITION=$?
#echo $CONDITION

while :; do

  until [ $CONDITION -eq 1 ] ; do
#     echo same
     sleep 5
     cmp /weather/data/goes16/TIRE/08/latest.nc /dev/shm/latest_TIRE_4panelwvir.nc > /dev/null
     CONDITION=$?
  done

#  echo different
  cp /weather/data/goes16/TIRE/08/latest.nc /dev/shm/latest_TIRE_4panelwvir.nc
  sleep 135
  /home/poker/miniconda3/envs/goes16_201710/bin/python goes16_conusc_wv_irc_fixeddisk.py /dev/shm/latest_TIRE_4panelwvir.nc
  cmp /weather/data/goes16/TIRE/08/latest.nc /dev/shm/latest_TIRE_4panelwvir.nc > /dev/null
  CONDITION=$?
#  echo repeat

done



python goes16_conusc_wvh.py $time


