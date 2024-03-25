#!/bin/bash

# Baselines perfomances
for encoding in standard split; do
    for train_ds in Zaragoza ILS Magnificat Guatemala Mottecta Primens; do
        python -u train.py --ds_name $train_ds --encoding_type $encoding
    done
done
# Synthetic over real -> OOD baselines
for encoding in standard split; do
    for test_ds in Zaragoza ILS Magnificat Guatemala Mottecta; do
        python -u test.py --train_ds_name Primens --test_ds_name $test_ds --encoding_type $encoding --checkpoint_path weights/Baseline-UpperBound/Primens_${encoding}.ckpt
    done
done
# Synthetic-to-Real Source-Free Adaptation 
for encoding in standard split; do
    for test_ds in Zaragoza ILS Magnificat Guatemala Mottecta; do
        python -u da_train_random_search.py --train_ds_name Primens --test_ds_name $test_ds --encoding_type $encoding --num_random_combinations 50
    done
done
