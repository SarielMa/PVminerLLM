# PVminerLLM
**PVminerLLM: Structured Extraction of Patient Voice from Patient-Generated Text using Large Language Models**

PVminerLLM is a framework for **structured extraction of Patient Voice (PV)** signals from patient-generated text using large language models (LLMs). The project focuses on identifying structured communication elements such as patient concerns, experiences, and contextual signals from patient-authored messages.

This repository contains the **training scripts, model preparation utilities, and evaluation pipeline** used in the PVminerLLM paper.

---

# Overview

The PVminerLLM pipeline includes three main components:

1. **Supervised Fine-Tuning (SFT)**  
   Large language models are fine-tuned using LoRA / QLoRA for structured extraction.

2. **Model Merging**  
   LoRA adapters are merged into the base models for inference and evaluation.

3. **Evaluation using FinBen**  
   Structured extraction performance is evaluated using the FinBen evaluation framework.

---

# Repository Structure

```
PVminerLLM/
│
├── sft_peft_ddp.py
│   Supervised fine-tuning script using PEFT (LoRA / QLoRA) with distributed training.
│
├── merge_lora.py
│   Utility to merge LoRA adapters into the base model.
│
├── sft_from_sft_to_finben.sh
│   End-to-end pipeline script for training and evaluation.
│
├── apply_server.sh
│   Example job script for running training on HPC clusters.
│
├── environment.yml
│   Conda environment configuration.
│
└── pv_utils.py
    Task-specific evaluation utilities used by FinBen.
```

---

# Pretrained Models

The supervised fine-tuned PVminerLLM models are available on Hugging Face:

- https://huggingface.co/lm2445/voice_70b_llama3.3_instruct
- https://huggingface.co/lm2445/voice_8b_llama3.1_instruct
- https://huggingface.co/lm2445/voice_3b_llama3.2_instruct
- https://huggingface.co/lm2445/voice_qwen2.5_1.5b_instruct

These models are fine-tuned for **structured extraction of Patient Voice signals**.

---

# Environment Setup

Create the conda environment using the provided configuration:

```bash
conda env create -f environment.yml
conda activate finben_vllm3
```

This environment includes dependencies for:

- PyTorch
- Hugging Face Transformers
- PEFT
- Datasets
- FinBen evaluation framework

---

# Training (Supervised Fine-Tuning)

The training script supports **multi-GPU distributed training**.

Example:

```bash
torchrun --nproc_per_node=2 sft_peft_ddp.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path <dataset_path> \
    --output_dir <output_dir>
```

Key features:

- PEFT-based LoRA / QLoRA fine-tuning
- prompt masking (loss applied only to answer tokens)
- large context length support
- distributed training with `torchrun`

---

# Merging LoRA Adapters

After training, LoRA adapters can be merged into the base model:

```bash
python merge_lora.py \
    --base meta-llama/Llama-3.1-8B-Instruct \
    --adapter <lora_adapter_path> \
    --out <merged_model_output>
```

This produces a **fully merged model** for inference and evaluation.

---

# Evaluation

Evaluation is performed using **FinBen**, a benchmark framework for evaluating LLMs.

Official repository:

https://github.com/The-FinAI/finlm_eval

For this project we provide a modified setup:

https://github.com/SarielMa/finben_modified

Follow the installation instructions in that repository to configure FinBen.

---

# PVminer Evaluation Task

The task-specific evaluation logic is implemented in:

```
pv_utils.py
```

This script defines the **PVminer structured extraction evaluation** used within FinBen.

---

# Running the Full Pipeline

An example end-to-end pipeline is provided:

```bash
sft_from_sft_to_finben.sh
```

This script performs:

1. Model fine-tuning  
2. LoRA adapter merging  
3. Evaluation using FinBen  

---

# HPC Training

Example HPC job configuration is provided in:

```
apply_server.sh
```

This demonstrates:

- multi-GPU training
- environment setup
- conda activation
- cluster job execution

---

# Citation

If you use PVminerLLM in your research, please cite:

```
PVminerLLM: Structured Extraction of Patient Voice from Patient-Generated Text using Large Language Models
```

---

<!-- # Contact

**Linhai Ma**  
Yale University School of Medicine -->
