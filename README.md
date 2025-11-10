# High-precision Flood Change Detection with Lightweight SAR Transformer Network and Context-aware Attention for Enriched-diverse and Complex Flooding Scenarios

## Overview

We are delighted to share that our paper, **"High-precision Flood Change Detection with Lightweight SAR Transformer Network and Context-aware Attention for Enriched-diverse and Complex Flooding Scenarios"**, has been successfully accepted by the **ISPRS Journal of Photogrammetry and Remote Sensing (ISPRS 2026)**. 


# AWCA-Net
![Figure3](https://github.com/user-attachments/assets/b72b81f6-6960-43fd-a072-164f0cf4c628)


## Training

To train the model, run the following command:

```bash
CUDA_VISIBLE_DEVICES='GPUID' python main.py   --model_name=AWCANet   --file_root='./file/data'   --batch_size=16   --img_size=256   --max_steps=20000
```

### Arguments:
- `--model_name`: Specifies the model architecture to use (in this case, AWCA-Net).
- `--file_root`: Path to the dataset root directory.
- `--batch_size`: Batch size for training.
- `--img_size`: Image size for input images.
- `--max_steps`: Number of training steps.

## Evaluation

To evaluate the model, use the following command:

```bash
CUDA_VISIBLE_DEVICES='GPUID' python eval.py
```

### Notes:
- This will run the evaluation using the pre-trained model on the test set.

## Prediction

To make predictions, execute the following:

```bash
CUDA_VISIBLE_DEVICES='GPUID' python evalvis.py
```

### Notes:
- This will generate visual results of the predictions for your input data.
