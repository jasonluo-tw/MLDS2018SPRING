#!/bin/bash

if [ -d "./models" ]; then
	# file exists
	echo "model exists"
else
	echo "model not exists"
	wget --no-check-certificate "https://www.dropbox.com/s/6lziyftl2ru5tj5/models.tar.gz?dl=1" -O models.tar.gz
	tar zxvf models.tar.gz
fi


python testing.py $1 $2
