#!/bin/bash
echo "downloading tsp concorde data into $1"
mkdir -p $1
gdown https://drive.google.com/uc?id=1tavTdBoyZcSN7CxKkuWQsk-I3mi74BJM -O "$1/old-tsp-data.tar.gz"
echo "extracting tsp data into $1"
tar -xzvf "$1/old-tsp-data.tar.gz" -C $1
echo "Done!"