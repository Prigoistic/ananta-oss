# Ananta: Scientific LLM Fine-tuning Pipeline

Ananta is a specialized scientific reasoning language model based on DeepSeek-Math-7B, fine-tuned for symbolic mathematics and scientific problem-solving. This repository contains the complete pipeline for data processing, fine-tuning, and deployment. Through this initiative we are trying to implement a new LLM architecture, heavily inspired from the Moosbeaur-Poole Algorithm which in short says that, multiply matrixes smarter.

## 🔬 Project Overview

Ananta focuses on:

- **Symbolic reasoning** and mathematical problem-solving
- **Block-level output generation** optimized for scientific contexts
- **Parameter-efficient fine-tuning** using LoRA on consumer GPUs
- **Clean academic codebase** with comprehensive documentation

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (RTX 3050 or better)
- 16GB+ RAM recommended
- DeepMind mathematics_dataset-v1.0 in txt format

### Installation

```bash
git clone https://github.com/yourusername/ananta-update.git
cd ananta-update
pip install -r requirements.txt
```

### Usage Pipeline

1. **Data Processing**: Convert raw dataset to training format

   ```bash
   python src/data/data_processor.py
   # or for simpler conversion
   python src/data/simple_data_converter.py
   ```

2. **Fine-tuning**: Train the model with LoRA

   ```bash
   python src/training/train_ananta.py
   # or for easier training
   python src/training/easy_train.py
   ```

3. **Evaluation**: Test model performance

   ```bash
   python src/evaluation/evaluate_model.py
   ```

4. **Demo Interface**: Launch Gradio interface
   ```bash
   python demos/app.py
   ```

5. **Complete Pipeline**: Run the entire workflow
   ```bash
   python src/run_pipeline.py
   ```

## 📊 Model Specifications

- **Base Model**: deepseek-ai/deepseek-math-7b
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Target Modules**: q_proj, v_proj
- **Training Configuration**:
  - Batch size: 1 (with gradient accumulation: 16)
  - Learning rate: 5e-5
  - Epochs: 3
  - Precision: FP16

## 🔧 Deployment Options

### Local Deployment

- Use `app.py` for local Gradio interface
- Load model in LM Studio for interactive testing

### Cloud Deployment

- **Hugging Face Spaces**: Upload to HF Spaces with Gradio
- **FastAPI**: Production API using `deploy/api_server.py`
- **Docker**: Containerized deployment (see `Dockerfile`)

## 📁 Project Structure

```
ananta-update/
├── src/                          # Source code
│   ├── training/                 # Training scripts
│   │   ├── train_ananta.py      # Main training script with LoRA
│   │   └── easy_train.py        # Simplified training script
│   ├── data/                     # Data processing
│   │   ├── data_processor.py    # Main dataset processor
│   │   ├── flexible_data_processor.py  # Flexible format handler
│   │   ├── simple_data_converter.py    # Simple converter
│   │   └── check_dataset.py     # Dataset structure diagnostic
│   ├── evaluation/               # Model evaluation
│   │   └── evaluate_model.py    # Evaluation metrics and benchmarks
│   ├── utils/                    # Utility functions
│   └── run_pipeline.py          # Complete pipeline orchestration
│
├── deployment/                   # Deployment configurations
│   ├── huggingface/             # HuggingFace Spaces deployment
│   │   └── deploy_hf_spaces.py # HF Spaces setup script
│   └── api/                     # API deployment (future)
│
├── demos/                        # Demo applications
│   └── app.py                   # Gradio demo interface
│
├── tests/                        # Testing scripts
│   └── test_ananta.py           # Model testing utilities
│
├── configs/                      # Configuration files
│   ├── train_config.json        # Training configuration
│   └── simple_requirements.txt  # Minimal requirements
│
├── docs/                         # Documentation
│   └── SIMPLE_README.md         # Simplified documentation
│
├── EBSL-Engine/                  # Engine components (future)
├── HMTT/                         # HMTT components (future)
├── RSL/                          # RSL components (future)
│
├── requirements.txt              # Main dependencies
└── README.md                     # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📚 References
- [Hybrid Math Text Tonkenizer](https://www.researchgate.net/publication/393773297_Bridging_the_Semantic_Gap_A_Hybrid_Math-Text_Tokenizer_for_Enhanced_Logical_Reasoning_in_Large_Language_Models)
- [A Critical Analysis of the Proposed Recursive Logic Subsystem for Self-Learning LLMs in Scientific Discovery](https://www.researchgate.net/publication/395473790_A_Critical_Analysis_of_the_Proposed_Recursive_Logic_Subsystem_for_Self-Learning_LLMs_in_Scientific_Discovery)
- [DeepSeek-Math Paper](https://arxiv.org/abs/2402.03300)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Mathematics Dataset](https://github.com/deepmind/mathematics_dataset)

## 📄 License

MIT License - see LICENSE file for details.
