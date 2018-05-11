#!/bin/bash

if [ -d "./models" ]; then
	# file exists
	echo "model exists"
else
	echo "model not exists"
	#wget --no-check-certificate "" -O models.tar.gz
fi

tar zxvf models.tar.gz

python testing.py $1 $2
