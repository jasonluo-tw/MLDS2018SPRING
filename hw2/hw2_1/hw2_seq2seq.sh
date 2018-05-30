#!/bin/bash

if [ -d "./models" ]; then
	# file exists
	echo "model exists"
else
	echo "model not exists"
	wget --no-check-certificate "https://www.dropbox.com/s/2t492mthzc9wt5x/models.tar.gz?dl=1" -O models.tar.gz
	tar zxvf models.tar.gz
fi


python testing.py $1 $2
