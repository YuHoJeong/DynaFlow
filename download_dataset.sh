#!/bin/bash
set -e

pip install -q gdown

mkdir -p datasets

gdown --folder https://drive.google.com/drive/folders/1trEvHNNH6dXo81EJtPC0muD1_e3L8LVO \
    -O datasets/

echo "Dataset ready at datasets/go1_locomotion/"
