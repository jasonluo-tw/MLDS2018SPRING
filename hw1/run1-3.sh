#!/bin/bash


if [ $1 -eq 1 ]
then
 echo 'plot shuffled random'
 cd hw1-3
 python plot_random.py
elif [ $1 -eq 2 ]
then
 echo 'Number of parameters v.s. Generalization'
 cd hw1-3
 python plot_cifar10_params.py 
elif [ $1 -eq 3 ]
then
 echo 'interpolation plot'
 cd hw1-3
 python plot_interpo.py
elif [ $1 -eq 31 ]
then
 echo "interpolation for different optimizer"
 cd hw1-3
 python plot_interpo_op.py
elif [ $1 -eq 4 ]
then
 echo 'plot model sensitivity'
 cd hw1-3
 python plot_sensi.py
elif [ $1 -eq 5 ]
then
 echo 'plot model sharpness'
 cd hw1-3
 python plot_sharp.py

else
 echo 'You choose the wrong number'
fi
