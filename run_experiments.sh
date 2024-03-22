#!/bin/bash

# Baselines perfomances
for encoding in standard split; do
    for train_ds in b-59-850 ILS Magnificat Guatemala Mottecta Primens; do
        if [ $train_ds == "Magnificat" ] && [ $encoding == "standard" ]; then
            python -u train.py --ds_name $train_ds --encoding_type $encoding --patience 30
        else
            python -u train.py --ds_name $train_ds --encoding_type $encoding
        fi
    done
done
# Synthetic over real -> OOD baselines
for encoding in standard split; do
    for test_ds in b-59-850 ILS Magnificat Guatemala Mottecta; do
        python -u test.py --train_ds_name Primens --test_ds_name $test_ds --encoding_type $encoding --checkpoint_path weights/Baseline-UpperBound/Primens_${encoding}.ckpt
    done
done
