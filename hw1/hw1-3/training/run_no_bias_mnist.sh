#!/bin/bash

batch_size=100

rm -f loss_acc_sharpness.csv

while [ $batch_size -le 10000 ]
do

if [ $batch_size -gt 2000 ] && [ $batch_size -le 5000 ]
then
     epochs=200
elif [ $batch_size -gt 5000 ];
then
     epochs=250
else
     epochs=100
fi

echo "batch size:" $batch_size
echo "epochs:" $epochs
python mnist_dnn_nobias.py -al True -bs $batch_size -ps $epochs
batch_size=$((batch_size + 500))
done

