#!/bin/bash

export PATH="/home/poker/miniconda3/bin:$PATH"

cd /home/poker/goes17/pacusc_4panel

#  /weather/data/goes16/"+prod_id+"/"+band+"/latest.nc
cp /weather/data/goes16/TIRW/08/latest.nc /dev/shm/latest_TIRW_4panelwvir.nc
cmp /weather/data/goes16/TIRW/08/latest.nc /dev/shm/latest_TIRW_4panelwvir.nc > /dev/null
CONDITION=$?
#echo $CONDITION

while :; do

  until [ $CONDITION -eq 1 ] ; do
#     echo same
     sleep 5
     cmp /weather/data/goes16/TIRW/08/latest.nc /dev/shm/latest_TIRW_4panelwvir.nc > /dev/null
     CONDITION=$?
  done

#  echo different
  cp /weather/data/goes16/TIRW/08/latest.nc /dev/shm/latest_TIRW_4panelwvir.nc
  sleep 135
  /home/poker/miniconda3/bin/python goes17_pacusc_wv_irc_fixeddisk.py /dev/shm/latest_TIRW_4panelwvir.nc
  cmp /weather/data/goes16/TIRW/08/latest.nc /dev/shm/latest_TIRW_4panelwvir.nc > /dev/null
  CONDITION=$?
#  echo repeat

done

