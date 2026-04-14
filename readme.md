# GEN-KnowRD: Reframing AI for Rare Disease Recognition

This document describes the Gen-KnowRD research workflow for inference and embedding fine-tuning, as well as the execution environments and hardware configurations used across HPC and Databricks platforms. Within this broader workflow, Qwen3 8B–based models are included as part of the implementation.

---

## Models Used

This project uses the following **Qwen3** models hosted on Hugging Face:

- **Qwen3-Embedding-8B**  
  https://huggingface.co/Qwen/Qwen3-Embedding-8B

- **Qwen3-Reranker-8B**  
  https://huggingface.co/Qwen/Qwen3-Reranker-8B

The Hugging Face model pages are the **authoritative and up-to-date sources** for:
- Environment requirements  
- Installation instructions  
- Example usage  
- Model configuration options  
- License and usage restrictions  

---

## Inference

### Purpose
Inference is used for:
- Dense embedding generation  
- Disease reranking using the Qwen3 embedding and reranker models
- Large-scale evaluation and analysis pipelines  

### Execution Environment

Inference can be executed on an **HPC environment** when GPU acceleration is required, and integrated with downstream analytics and data processing workflows on **Databricks**.

Databricks is used for:
- Large-scale data preparation  
- PySpark- and Spark SQL–based analytics
  - BM25 (Okapi BM25) for sparse lexical retrieval and initial candidate ranking
  - RRF (Reciprocal Rank Fusion) for combining multiple ranking signals into a unified score
- Evaluation and result aggregation  

### Hardware Configuration

**HPC (Inference):**
- 2 × NVIDIA H100 GPUs  
- CUDA-enabled environment  
- Compatible PyTorch and driver versions  

**Databricks (Analytics & Integration):**
- 8 vCPUs  
- 64 GiB memory  
- 1–4 worker nodes (autoscaling enabled)

---

## Embedding Fine-Tuning

### Purpose
Embedding fine-tuning adapts **Qwen3-Embedding-8B** to LLM-generated rare disease sections to improve:
- Semantic similarity  
- Retrieval quality  
- Downstream reranking performance  

### Fine-Tuning Framework

Embedding fine-tuning is performed using **SWIFT**.

Official SWIFT documentation:  
https://swift.readthedocs.io/en/latest/GetStarted/SWIFT-installation.html

SWIFT supports:
- Distributed training  
- LoRA / parameter-efficient fine-tuning  
- Checkpointing and output management  

---

## HPC Training Setup

### Training Script

Embedding fine-tuning is configured to run on an **HPC cluster** using SLURM.

- **Training script:**  
  `HPC_scripts/embedding_training.slurm`

This script includes:
- GPU, memory, and wall-time configuration  
- SWIFT training command  
- Model checkpoint paths  
- Dataset input and output directories  

Users should review and adapt the SLURM configuration to match their local HPC environment.

### Hardware Configuration (HPC)

- 2 × NVIDIA H100 GPUs  
- CUDA-enabled environment  
- Compatible PyTorch and driver versions  

---

## Execution Scope Summary

| Task | Environment | Hardware |
|-----|------------|----------|
| Inference | HPC | 2 × NVIDIA H100 GPUs |
| Embedding Fine-Tuning | HPC (SLURM) | 2 × NVIDIA H100 GPUs |
| Data Processing & Evaluation | Databricks | 8 vCPUs, 64 GiB RAM, 1–4 workers |

---

## Notes and Best Practices

- These are **8B-parameter models** and require GPU resources for training and high-throughput inference.
- Verify CUDA, PyTorch, and driver compatibility before execution.
- Review all **model licenses** and **data usage restrictions** before training or deployment.

---

## Citation

Yan C, Su WC, Xin Y, Grabowska ME, Kerchberger VE, Borza VA, Wang J, Wang L, Li R, Lynn J, Dickson AL. GEN-KnowRD: Reframing AI for Rare Disease Recognition. medRxiv. 2026:2026-03.

## License / Intended Use
This repository is provided solely for non-commercial research purposes. It is intended to support academic research, evaluation, and educational use only.
Use of this repository, in whole or in part, for commercial purposes, clinical deployment, product development, or other business use is not permitted without prior written permission from the authors.
