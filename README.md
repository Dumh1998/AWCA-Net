# AWCA-Net

# Data structure
train
├─train/A
├─train/B
└─train/label

test
├─test/A
├─test/B
└─test/label

A: images of t1 phase;
B: images of t2 phase;
label: label maps;

# Training
CUDA_VISIBLE_DEVICES='GPUID' python main.py
--model_name=AWCANet
--file_root='./file/data'
--batch_size=16
--img_size=256
--max_steps=20000

# Evaluate
CUDA_VISIBLE_DEVICES='GPUID' python eval.py

# Predict
CUDA_VISIBLE_DEVICES='GPUID' python evalvis.py
