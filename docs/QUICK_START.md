# Ananta Quick Start Guide

This guide will help you get started with Ananta quickly.

## Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (recommended: RTX 3050 or better)
- 16GB+ RAM
- DeepMind mathematics_dataset-v1.0 (txt format)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Prigoistic/ananta-update.git
cd ananta-update
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 3. Install dependencies

For full installation:
```bash
pip install -r requirements.txt
```

For minimal installation:
```bash
pip install -r configs/simple_requirements.txt
```

For editable installation (development):
```bash
pip install -e .
```

## Usage

### Step 1: Prepare Your Data

Place your DeepMind mathematics dataset in the project directory, then run:

```bash
# Check your dataset structure
python src/data/check_dataset.py

# Convert dataset to training format
python src/data/data_processor.py
```

For simpler conversion:
```bash
python src/data/simple_data_converter.py
```

### Step 2: Train the Model

```bash
# Full training with all features
python src/training/train_ananta.py

# or simple training
python src/training/easy_train.py
```

The training will:
- Load DeepSeek-Math-7B base model
- Apply LoRA fine-tuning
- Save checkpoints every 500 steps
- Create logs for monitoring

### Step 3: Evaluate the Model

```bash
python src/evaluation/evaluate_model.py --model_path ./deepseek_finetuned
```

### Step 4: Try the Demo

```bash
python demos/app.py
```

This will launch a Gradio interface where you can interact with your trained model.

## Running the Complete Pipeline

To run all steps automatically:

```bash
python src/run_pipeline.py
```

## Testing Your Model

Quick test without the full demo:

```bash
python tests/test_ananta.py
```

## Configuration

Edit `configs/train_config.json` to customize:
- Model parameters (LoRA rank, alpha, dropout)
- Training hyperparameters (batch size, learning rate, epochs)
- Data settings (dataset path, validation split)
- Monitoring options (wandb, tensorboard)

## Deployment

### Hugging Face Spaces

```bash
python deployment/huggingface/deploy_hf_spaces.py
```

Follow the prompts to deploy your model to Hugging Face Spaces.

## Troubleshooting

### Out of Memory Errors

1. Reduce batch size in `configs/train_config.json`
2. Enable 8-bit quantization (already enabled by default)
3. Reduce `max_seq_length` in configuration

### Dataset Not Found

Run `python src/data/check_dataset.py` to diagnose the issue.

### Import Errors

Make sure you're in the project root directory and have activated your virtual environment.

## Directory Structure

- `src/` - Source code
  - `training/` - Training scripts
  - `data/` - Data processing
  - `evaluation/` - Model evaluation
  - `utils/` - Utilities
- `configs/` - Configuration files
- `demos/` - Demo applications
- `tests/` - Testing scripts
- `deployment/` - Deployment utilities
- `docs/` - Documentation

## Getting Help

- Check `docs/SIMPLE_README.md` for additional documentation
- Review configuration in `configs/train_config.json`
- Check logs in project root directory

## Next Steps

1. Experiment with different hyperparameters
2. Try different subsets of the mathematics dataset
3. Evaluate on custom test cases
4. Deploy to cloud platforms
5. Integrate with your applications

Happy training! 🚀
