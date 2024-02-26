#!/bin/bash

# - MENSURAL
for encoding in standard split; do
    for train_ds in b-59-850 ILS Magnificat Guatemala Mottecta; do
        python -u train.py --ds_name $train_ds --encoding_type $encoding
        for test_ds in b-59-850 ILS Magnificat Guatemala Mottecta; do
            if [ $train_ds != $test_ds ]; then
                python -u test.py --train_ds_name $train_ds --test_ds_name $test_ds --checkpoint_path weights/Baseline-UpperBound/$train_ds\_$encoding.ckpt --encoding_type $encoding
                python -u da_train_random_search.py --train_ds_name $train_ds --test_ds_name $test_ds --encoding_type $encoding --num_random_combinations 50
            fi
        done
    done
done
# - MODERN
for encoding in standard split; do
    for train_ds in Primus CameraPrimus FMT Malaga; do
        python -u train.py --ds_name $train_ds --encoding_type $encoding
        for test_ds in Primus CameraPrimus FMT Malaga; do
            if [ $train_ds != $test_ds ]; then
                if [ $train_ds == Primus ] && [ $test_ds == CameraPrimus ]; then
                    continue
                fi
                if [ $train_ds == CameraPrimus ] && [ $test_ds == Primus ]; then
                    continue
                fi
                python -u test.py --train_ds_name $train_ds --test_ds_name $test_ds --checkpoint_path weights/Baseline-UpperBound/$train_ds\_$encoding.ckpt --encoding_type $encoding
                python -u da_train_random_search.py --train_ds_name $train_ds --test_ds_name $test_ds --encoding_type $encoding --num_random_combinations 50
            fi
        done
    done
done