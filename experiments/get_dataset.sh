#!/bin/bash

DATASET=$1

wget https://zenodo.org/records/13764675/files/$DATASET.tar.gz
echo "extracting data ..."
tar -xvf $DATASET.tar.gz

rm $DATASET.tar.gz