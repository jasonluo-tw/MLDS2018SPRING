#!/bin/bash

params=1

while [ $params -le 10 ]
do
 echo "parameter:" $params
 python keras_cifar_10.py $params $params

 params=$((params + 2))
done 
