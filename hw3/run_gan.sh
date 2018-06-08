#!/bin/bash

if [ -d "./models" ]; then
	# file exists
	echo "model exists"
else
	echo "model not exists"
	wget --no-check-certificate ""
	tar zxvf models.tar.gz
fi

python test.py
