#!/bin/bash

rm -r test/
python src/batch_hpc_fast.py example/smt_gal4.csv -o test/ \
    --seed 1 --bit-depth 16 \
    --save-positions \
    --smt --smt-focus "synthetic" --smt-background 1 --smt-photons 5000\
    --near-radius-nm 200 \
    --trajectory-stride 10 


    