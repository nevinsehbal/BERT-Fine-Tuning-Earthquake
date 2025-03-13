!/bin/bash

# LEARNING RATE Hyperparameter Tuning: 3e-5, 2e-5, 5e-5, 1e-4, 1e-5
/home/hekimoglu/workspace/git/BERT-Fine-Tuning-Earthquake/bert-env/bin/python\
 /home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/src/final-stage3.py \
    --model_path "/home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/logs_second_stage/forecasts_1738822520_epochs50_patch50_batch8_sample600_lr5e-05_modelpatches1737315105.74911/final_model.pt" \
    --epochs 50 \
    --learning_rate 0.00003 \
    --batch_size 8 \
    --patch_size 50 \
    --sample_length 600

/home/hekimoglu/workspace/git/BERT-Fine-Tuning-Earthquake/bert-env/bin/python\
 /home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/src/final-stage3.py \
    --model_path "/home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/second_stage_logs/forecasts1737317559/final_model.pt" \
    --epochs 50 \
    --learning_rate 0.00002 \
    --batch_size 8 \
    --patch_size 50 \
    --sample_length 600

/home/hekimoglu/workspace/git/BERT-Fine-Tuning-Earthquake/bert-env/bin/python\
 /home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/src/final-stage3.py \
    --model_path "/home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/second_stage_logs/forecasts1737317559/final_model.pt" \
    --epochs 50 \
    --learning_rate 0.00005 \
    --batch_size 8 \
    --patch_size 50 \
    --sample_length 600

/home/hekimoglu/workspace/git/BERT-Fine-Tuning-Earthquake/bert-env/bin/python\
 /home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/src/final-stage3.py \
    --model_path "/home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/second_stage_logs/forecasts1737317559/final_model.pt" \
    --epochs 50 \
    --learning_rate 0.0001 \
    --batch_size 8 \
    --patch_size 50 \
    --sample_length 600

/home/hekimoglu/workspace/git/BERT-Fine-Tuning-Earthquake/bert-env/bin/python\
 /home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/src/final-stage3.py \
    --model_path "/home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/second_stage_logs/forecasts1737317559/final_model.pt" \
    --epochs 50 \
    --learning_rate 0.00001 \
    --batch_size 8 \
    --patch_size 50 \
    --sample_length 600