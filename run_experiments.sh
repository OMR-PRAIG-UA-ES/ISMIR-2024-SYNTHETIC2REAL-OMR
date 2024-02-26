#!/bin/bash

# - MENSURAL: Baselines perfomances
for encoding in standard split; do
    for train_ds in b-59-850 ILS Magnificat Guatemala Mottecta; do
        python -u train.py --ds_name $train_ds --encoding_type $encoding
    done
done