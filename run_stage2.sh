!/bin/bash

# LEARNING RATE Hyperparameter Tuning: 3e-5, 2e-5, 5e-5, 1e-4, 1e-5

# Chosen 5e-5 
/home/hekimoglu/workspace/git/BERT-Fine-Tuning-Earthquake/bert-env/bin/python\
 /home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/src/final-stage2.py \
    --model_path "/home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/logs/patches1737315105.74911/final_model.pt" \
    --epochs 50 \
    --learning_rate 0.00003 \
    --batch_size 8 \
    --patch_size 50 \
    --sample_length 600

/home/hekimoglu/workspace/git/BERT-Fine-Tuning-Earthquake/bert-env/bin/python\
 /home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/src/final-stage2.py \
    --model_path "/home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/logs/patches1737315105.74911/final_model.pt" \
    --epochs 50 \
    --learning_rate 0.00002 \
    --batch_size 8 \
    --patch_size 50 \
    --sample_length 600

/home/hekimoglu/workspace/git/BERT-Fine-Tuning-Earthquake/bert-env/bin/python\
 /home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/src/final-stage2.py \
    --model_path "/home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/logs/patches1737315105.74911/final_model.pt" \
    --epochs 50 \
    --learning_rate 0.00005 \
    --batch_size 8 \
    --patch_size 50 \
    --sample_length 600

/home/hekimoglu/workspace/git/BERT-Fine-Tuning-Earthquake/bert-env/bin/python\
 /home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/src/final-stage2.py \
    --model_path "/home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/logs/patches1737315105.74911/final_model.pt" \
    --epochs 50 \
    --learning_rate 0.0001 \
    --batch_size 8 \
    --patch_size 50 \
    --sample_length 600

/home/hekimoglu/workspace/git/BERT-Fine-Tuning-Earthquake/bert-env/bin/python\
 /home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/src/final-stage2.py \
    --model_path "/home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/logs/patches1737315105.74911/final_model.pt" \
    --epochs 50 \
    --learning_rate 0.00001 \
    --batch_size 8 \
    --patch_size 50 \
    --sample_length 600

# # BATCH SIZE Hyperparameter Tuning: 2, 4, 8, 16, 32
# /home/hekimoglu/workspace/git/BERT-Fine-Tuning-Earthquake/bert-env/bin/python\
#  /home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/src/final-stage2.py \
#     --model_path "/home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/logs/patches1737315105.74911/final_model.pt" \
#     --epochs 50 \
#     --learning_rate 0.00003 \ # 3e-5
#     --batch_size 8 \
#     --patch_size 50 \
#     --sample_length 600

# /home/hekimoglu/workspace/git/BERT-Fine-Tuning-Earthquake/bert-env/bin/python\
#  /home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/src/final-stage2.py \
#     --model_path "/home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/logs/patches1737315105.74911/final_model.pt" \
#     --epochs 50 \
#     --learning_rate 0.00002 \ # 2e-5
#     --batch_size 8 \
#     --patch_size 50 \
#     --sample_length 600

# /home/hekimoglu/workspace/git/BERT-Fine-Tuning-Earthquake/bert-env/bin/python\
#  /home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/src/final-stage2.py \
#     --model_path "/home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/logs/patches1737315105.74911/final_model.pt" \
#     --epochs 50 \
#     --learning_rate 0.00005 \ # 5e-5
#     --batch_size 8 \
#     --patch_size 50 \
#     --sample_length 600

# /home/hekimoglu/workspace/git/BERT-Fine-Tuning-Earthquake/bert-env/bin/python\
#  /home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/src/final-stage2.py \
#     --model_path "/home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/logs/patches1737315105.74911/final_model.pt" \
#     --epochs 50 \
#     --learning_rate 0.0001 \ # 1e-4
#     --batch_size 8 \
#     --patch_size 50 \
#     --sample_length 600

# /home/hekimoglu/workspace/git/BERT-Fine-Tuning-Earthquake/bert-env/bin/python\
#  /home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/src/final-stage2.py \
#     --model_path "/home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/logs/patches1737315105.74911/final_model.pt" \
#     --epochs 50 \
#     --learning_rate 0.00001 \ # 1e-5
#     --batch_size 8 \
#     --patch_size 50 \
#     --sample_length 600