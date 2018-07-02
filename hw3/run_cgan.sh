#!/bin/bash

if [ -d "./models2" ]; then
	# file exists
	echo "model exists"
else
	echo "model not exists"
	wget --no-check-certificate "https://www.dropbox.com/s/b7v6r96hv2nnt4l/models.tar.gz?dl=0" -O models.tar.gz
	tar zxvf models.tar.gz
fi

python testc.py $1
