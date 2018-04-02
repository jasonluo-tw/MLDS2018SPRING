#!/bin/bash

batch_size=32

rm -f loss_acc_seni.csv

while [ $batch_size -le 12000 ]
do
echo "batch size:" $batch_size
python mnist_dnn.py -al True -bs $batch_size
batch_size=$((batch_size * 2))
done
