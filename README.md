# Abstractive Question Answering with Qwen3

## Overview
This project implements an abstractive question answering system using the Qwen3 language model. The system is fine-tuned using QLoRA (Quantized Low-Rank Adaptation) for efficient training and generates free-form text answers based on provided context.

## Key Features
- **Efficient Fine-tuning**: Utilizes QLoRA with 4-bit quantization for memory-efficient model adaptation
- **Optimized Performance**: Achieves a 0.799 combined score (SAS × EM) on the validation set
- **Semantic Evaluation**: Implements both semantic similarity (SAS) and exact match metrics
- **Memory Efficient**: Uses Unsloth for optimized training, making it suitable for consumer GPUs

## Technical Implementation

### Model Architecture
- Base Model: Qwen3-8B (via Unsloth)
- Fine-tuning Method: QLoRA (4-bit quantization)
- LoRA Configuration:
  - Rank: 64
  - Alpha: 64
  - Target Modules: Query, Key, Value, Output, Gate, Up/Down projections

### Training Configuration
- Epochs: 1
- Learning Rate: 2e-4
- Scheduler: Linear decay
- Batch Size: 2 (per device)
- Gradient Accumulation Steps: 4
- Effective Batch Size: 8

### Generation Parameters
- Max New Tokens: 400
- Temperature: 0.6
- Top-p: 0.9
- Sampling Strategy: Nucleus sampling

## Performance Metrics
- Semantic Answer Similarity (SAS): 0.9686
- Exact Match (EM): 0.8250
- Combined Score (SAS × EM): 0.7991

## Requirements
Key dependencies include:
```
unsloth[colab-new]
simple-llms-eval
torch
transformers
datasets
peft
trl
accelerate
bitsandbytes
```
Full requirements are available in `requirements.txt`.

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the training script:
```bash
python train.py
```

The script will:
- Load and prepare the dataset
- Fine-tune the model using QLoRA
- Evaluate on the validation set
- Generate predictions for the test set

## Technical Notes

### Memory Optimization
- Uses 4-bit quantization for the base model
- Implements Unsloth's optimized gradient checkpointing
- Efficient token generation with proper cache handling

### Data Processing
- Implements chat template formatting for instruction tuning
- Handles conversation history in a structured format
- Cleans generated outputs using regex patterns

### Evaluation
- Uses sentence-transformers for semantic similarity scoring
- Implements case-insensitive exact matching
- Combines semantic and exact metrics for comprehensive evaluation

## Results
The model demonstrates strong performance in both semantic understanding (SAS: 0.962) and exact matching (EM: 0.815), indicating its capability to generate both accurate and contextually appropriate responses.

## Future Improvements
- Experiment with different base models (e.g., Llama3, Gemma)
- Implement beam search for potentially more precise generation
- Explore multi-epoch training with learning rate optimization
- Add support for streaming responses
- Implement batched inference for improved throughput

## License
MIT License - feel free to use and modify as needed. 
