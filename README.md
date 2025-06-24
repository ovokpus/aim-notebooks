# AI Makerspace Notebooks

This repository contains comprehensive Jupyter notebooks covering the complete spectrum of Large Language Model (LLM) development - from Python fundamentals and API usage to advanced fine-tuning and Reinforcement Learning from Human Feedback (RLHF).

## 📚 Contents

### **Foundation Level**

### 1. **llm_python_basics_extended.ipynb** - Python Fundamentals for LLM Development
A comprehensive foundation course covering essential Python concepts for LLM development.

**Key Features:**
- **Target Audience**: Beginners to intermediate Python users
- **Focus**: Python fundamentals with LLM context
- **Topics**: Variables, control flow, functions, classes, async programming
- **Libraries**: NumPy, Pandas, OpenAI API basics
- **Activities**: 10+ hands-on coding exercises

**Learning Path:**
- Variables and data types with type hints
- Control flow and loops for automation
- Functions with docstrings and type annotations
- Object-oriented programming for LLM applications
- Exception handling and async programming
- First OpenAI API calls and response handling

### 2. **python_openai.ipynb** - OpenAI API Exploration
Practical exploration of OpenAI's API capabilities across multiple modalities.

**Key Features:**
- **API Coverage**: Chat completions, code generation, text summarization, image generation
- **Tasks**: 6 comprehensive tasks covering different API endpoints
- **Libraries**: OpenAI, Pandas, tiktoken, PyPDF, matplotlib
- **Applications**: Email generation, code creation, document processing, DALL-E image generation

**Core Tasks:**
1. Module imports and API setup
2. Email generation from product reviews
3. Python code generation for algorithms
4. Text summarization with token counting
5. Research paper summarization
6. Image generation with DALL-E 3

### 3. **Accessing_GPT_4_1_nano_Like_a_Developer.ipynb** - Developer-Focused GPT-4 Usage
Professional approach to working with GPT-4.1-nano API with advanced prompting techniques.

**Key Features:**
- **Model**: GPT-4.1-nano access and optimization
- **Focus**: Developer workflows and best practices
- **Techniques**: Few-shot prompting, chain-of-thought reasoning
- **Tools**: Helper functions, response formatting, role management

**Advanced Concepts:**
- System, assistant, and user role management  
- Few-shot learning with custom vocabulary
- Chain-of-thought prompting for complex reasoning
- Response parsing and professional API integration

### **Advanced Level**

### 4. **llm_fine-tuning_with_qlora.ipynb** - LLM Fine-Tuning with QLoRA
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

### 5. **Reward_Model_and_PPO_Training_RLHF_in_Practice.ipynb** - RLHF Implementation
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
- OpenAI API key (for foundation notebooks)
- CUDA-compatible GPU (A100 recommended for advanced notebooks)
- Google Colab Pro account (recommended for intensive training)

### Recommended Learning Path
1. **Start with Foundation**: `llm_python_basics_extended.ipynb` → `python_openai.ipynb` → `Accessing_GPT_4_1_nano_Like_a_Developer.ipynb`
2. **Advance to Training**: `llm_fine-tuning_with_qlora.ipynb` → `Reward_Model_and_PPO_Training_RLHF_in_Practice.ipynb`

### Installation
The notebooks include installation cells for all required dependencies:

**Foundation Notebooks:**
```bash
pip install openai pandas tiktoken pypdf matplotlib numpy requests ipython
```

**Advanced Training Notebooks:**
```bash
pip install -qU transformers accelerate bitsandbytes peft trl datasets tqdm
```

### Running the Notebooks

#### **Foundation Notebooks** (No GPU Required):

**Python Basics:**
1. Open `llm_python_basics_extended.ipynb`
2. Work through 11 comprehensive sections
3. Complete 10+ hands-on activities
4. Practice with NumPy, Pandas, and basic OpenAI calls

**OpenAI API Exploration:**
1. Open `python_openai.ipynb`  
2. Set your OpenAI API key
3. Complete 6 tasks covering different API endpoints
4. Experiment with chat, code generation, summarization, and image creation

**Developer GPT-4 Usage:**
1. Open `Accessing_GPT_4_1_nano_Like_a_Developer.ipynb`
2. Set up OpenAI API credentials
3. Master advanced prompting techniques
4. Learn few-shot and chain-of-thought prompting

#### **Advanced Notebooks** (GPU Required):

**LLM Fine-Tuning with QLoRA:**
1. Open `llm_fine-tuning_with_qlora.ipynb` in Google Colab Pro
2. Ensure you have an A100 GPU runtime selected
3. Set up your Hugging Face token for model uploads
4. Follow the 5 main tasks:
   - Task 1: Model Loading with Quantization
   - Task 2: Data Preparation 
   - Task 3: PEFT LoRA Configuration
   - Task 4: Model Training
   - Task 5: Model Sharing on Hugging Face Hub

**RLHF Training:**
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

### **Foundation Skills**
1. **Python for LLM Development**
   - Modern Python practices with type hints
   - Object-oriented programming for ML applications
   - Async programming and exception handling
   - Data manipulation with NumPy and Pandas

2. **API Integration & Usage**
   - OpenAI API authentication and configuration
   - Multi-modal API calls (text, code, images)
   - Token counting and cost optimization
   - Response parsing and error handling

3. **Advanced Prompting Techniques**
   - System, assistant, and user role management
   - Few-shot learning implementation
   - Chain-of-thought reasoning
   - Professional prompt engineering

### **Advanced ML Skills**
4. **Parameter-Efficient Fine-Tuning**
   - QLoRA implementation and theory
   - Memory optimization techniques
   - Adapter-based training strategies

5. **RLHF Implementation**
   - Reward model training from human preferences
   - PPO algorithm for LLM alignment
   - Safety and alignment evaluation methods

6. **Production ML Skills**
   - Model quantization for deployment
   - Hugging Face Hub integration
   - GPU memory optimization
   - Evaluation and benchmarking

## 🔧 Hardware Requirements

### **Foundation Notebooks** (CPU-Based)
- **GPU**: Not required
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 5GB+ for dependencies and outputs
- **Network**: Stable internet for API calls

### **Advanced Training Notebooks** (GPU-Required)

**Minimum Requirements:**
- **GPU**: T4 (with reduced batch sizes and model parameters)
- **RAM**: 16GB+
- **Storage**: 50GB+ for model downloads and checkpoints

**Recommended Requirements:**
- **GPU**: A100 40GB (for full functionality)
- **RAM**: 32GB+
- **Storage**: 100GB+ for optimal performance

## 📈 Expected Deliverables

### **Foundation Level**
**Python Basics Completion:**
- Mastery of 10+ hands-on Python exercises
- Understanding of LLM-focused programming patterns
- First successful OpenAI API integration

**API Proficiency:**
- Multi-modal API implementations (text, code, image)
- Professional prompt engineering skills
- Understanding of token economics and optimization

**Developer Skills:**
- Advanced prompting technique mastery
- Chain-of-thought and few-shot learning implementation
- Production-ready API integration patterns

### **Advanced Level**
**QLoRA Fine-Tuning:**
- Fine-tuned model uploaded to Hugging Face Hub
- Demonstration of improved legal document summarization
- Understanding of LoRA efficiency vs. full fine-tuning

**RLHF Project:**
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