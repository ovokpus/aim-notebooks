# AI Makerspace Notebooks

This repository contains comprehensive Jupyter notebooks for advanced Large Language Model (LLM) training techniques, including fine-tuning with LoRA and Reinforcement Learning from Human Feedback (RLHF).

## 📚 Contents

### 1. **LLME_Challenge_staged.ipynb** - LLM Fine-Tuning Challenge
A complete end-to-end tutorial for fine-tuning large language models using Parameter-Efficient Fine-Tuning (PEFT) techniques.

**Key Features:**
- **Model**: Fine-tuning `NousResearch/Meta-Llama-3.1-8B-Instruct` 
- **Task**: Legal document summarization
- **Technique**: QLoRA (Quantized Low-Rank Adaptation)
- **Dataset**: Plain English Summary of Contracts
- **Hardware**: Optimized for Google Colab Pro (A100 GPU)

**Learning Objectives:**
- Understanding quantization and 4-bit training
- Implementing LoRA adapters for efficient fine-tuning
- Working with legal document summarization datasets
- Model deployment to Hugging Face Hub

### 2. **Reward_Model_and_PPO_Training_RLHF_in_Practice.ipynb** - RLHF Implementation
A practical implementation of Reinforcement Learning from Human Feedback for improving model safety and alignment.

**Key Features:**
- **Base Model**: `meta-llama/Llama-3.1-8B-Instruct`
- **Dataset**: Anthropic's `hh-rlhf` for reward modeling
- **Training Pipeline**: Reward Model → PPO Training
- **Focus**: Reducing harmful outputs and improving helpfulness
- **Hardware**: Requires A100 GPU

**RLHF Pipeline:**
1. **Reward Model Training**: Train a preference model using human feedback data
2. **PPO Training**: Use Proximal Policy Optimization to align the LLM with the reward model
3. **Evaluation**: Benchmark model performance on toxicity and helpfulness metrics

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (A100 recommended for full functionality)
- Google Colab Pro account (recommended)

### Installation
The notebooks include installation cells for all required dependencies:
```bash
pip install -qU transformers accelerate bitsandbytes peft trl datasets tqdm
```

### Running the Notebooks

#### For LLM Fine-Tuning (LLME Challenge):
1. Open `LLME_Challenge_staged.ipynb` in Google Colab Pro
2. Ensure you have an A100 GPU runtime selected
3. Set up your Hugging Face token for model uploads
4. Follow the 5 main tasks:
   - Task 1: Model Loading with Quantization
   - Task 2: Data Preparation 
   - Task 3: PEFT LoRA Configuration
   - Task 4: Model Training
   - Task 5: Model Sharing on Hugging Face Hub

#### For RLHF Training:
1. Open `Reward_Model_and_PPO_Training_RLHF_in_Practice.ipynb`
2. Ensure A100 GPU availability
3. Complete the full RLHF pipeline:
   - Load base model and evaluation datasets
   - Train reward model on preference data
   - Implement PPO training loop
   - Evaluate model improvements

## 🛠 Technical Details

### Quantization Strategy
- **4-bit Quantization**: Using NF4 (Normal Float 4) for optimal memory efficiency
- **Double Quantization**: Additional compression of quantization constants
- **Block-wise Quantization**: Exploiting normal distribution of model weights

### LoRA Configuration
- **Rank (r)**: 16 (configurable based on performance needs)
- **Alpha**: 32 (typically 2 × rank)
- **Dropout**: 0.1
- **Target Modules**: All linear layers
- **Task Type**: Causal Language Modeling

### Training Parameters
- **Batch Size**: 1 per device (gradient accumulation used)
- **Learning Rate**: 2e-4 (higher than standard due to LoRA efficiency)
- **Max Steps**: 500 (adjustable based on dataset size)
- **Evaluation Strategy**: Every 25 steps

## 📊 Datasets

### Legal Summarization Dataset
- **Source**: Plain English Summary of Contracts
- **Size**: ~1000 legal document-summary pairs
- **Format**: EULA, TOS, and other legal documents with human-readable summaries
- **Task**: Convert complex legal language into accessible summaries

### Anthropic HH-RLHF Dataset
- **Source**: Anthropic's human preference dataset
- **Purpose**: Training reward models for RLHF
- **Format**: Chosen vs. rejected response pairs
- **Focus**: Helpfulness and harmlessness alignment

## 🎯 Key Learning Outcomes

By working through these notebooks, you will learn:

1. **Parameter-Efficient Fine-Tuning**
   - QLoRA implementation and theory
   - Memory optimization techniques
   - Adapter-based training strategies

2. **RLHF Implementation**
   - Reward model training from human preferences
   - PPO algorithm for LLM alignment
   - Safety and alignment evaluation methods

3. **Production ML Skills**
   - Model quantization for deployment
   - Hugging Face Hub integration
   - GPU memory optimization
   - Evaluation and benchmarking

## 🔧 Hardware Requirements

### Minimum Requirements
- **GPU**: T4 (with reduced batch sizes and model parameters)
- **RAM**: 16GB+
- **Storage**: 50GB+ for model downloads and checkpoints

### Recommended Requirements
- **GPU**: A100 40GB (for full functionality)
- **RAM**: 32GB+
- **Storage**: 100GB+ for optimal performance

## 📈 Expected Deliverables

### LLME Challenge
- Fine-tuned model uploaded to Hugging Face Hub
- Demonstration of improved legal document summarization
- Understanding of LoRA efficiency vs. full fine-tuning

### RLHF Project
- Trained reward model for safety evaluation
- PPO-optimized model with reduced toxicity
- Comparative analysis of pre/post RLHF performance

## 🤝 Contributing

This repository is part of the AI Makerspace educational curriculum. If you find issues or have suggestions for improvements:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed descriptions

## 📝 License

Educational use - please refer to individual model licenses:
- Llama 3.1 models are subject to Meta's custom license
- Datasets may have their own usage restrictions
- Code examples are provided for educational purposes

## 🔗 Additional Resources

- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
- [QLoRA Paper](https://arxiv.org/pdf/2305.14314.pdf)
- [LoRA Paper](https://arxiv.org/pdf/2106.09685.pdf)
- [Anthropic RLHF Paper](https://arxiv.org/abs/2204.05862)
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)

---

**Note**: These notebooks are designed for educational purposes and require significant computational resources. Colab Pro with A100 access is recommended for the full experience.