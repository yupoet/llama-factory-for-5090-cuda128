# LLaMA-Factory for CUDA 12.8+ (RTX 5090 / RTX PRO6000)

# LLaMA-Factory 适用于 CUDA 12.8+ (RTX 5090 / RTX PRO6000)

[English](#english) | [中文](#中文)

---

<a name="english"></a>
## 🇺🇸 English

### Introduction

A ready-to-use AutoDL image for LLaMA-Factory, specifically optimized for **NVIDIA RTX 5090** and **RTX PRO6000** GPUs with **CUDA 12.8** support. This image comes pre-configured with all necessary dependencies, including Flash Attention 2, DeepSpeed, and quantization tools.

### ✨ Features

- 🚀 **Blackwell Architecture Support** - Full support for RTX 5090/PRO6000 (SM 100)
- ⚡ **Flash Attention 2** - Pre-installed for maximum training efficiency
- 🔧 **DeepSpeed Integration** - Ready for distributed training
- 📦 **Quantization Tools** - BitsAndBytes for 4-bit/8-bit training
- 🌏 **China Mirror Pre-configured** - Fast downloads within mainland China

### 📦 Environment Specifications

#### System Environment

| Component | Version | Description |
|-----------|---------|-------------|
| **OS** | Ubuntu 22.04 | Base operating system |
| **Python** | 3.12 | Python runtime |
| **CUDA** | 12.8 | NVIDIA CUDA Toolkit |
| **cuDNN** | 9.x | Deep Neural Network library |

#### Core Dependencies

| Component | Version | Description |
|-----------|---------|-------------|
| **PyTorch** | 2.8.0+cu128 | Deep learning framework |
| **LLaMA-Factory** | 0.9.3 | Fine-tuning framework |
| **Transformers** | 4.52.4 | Hugging Face Transformers |
| **PEFT** | 0.15.2 | Parameter-Efficient Fine-Tuning |
| **TRL** | 0.9.6 | Transformer Reinforcement Learning |
| **Accelerate** | 1.7.0 | Hugging Face Accelerate |

#### Acceleration & Optimization

| Component | Version | Description |
|-----------|---------|-------------|
| **Flash Attention** | 2.8.3 | Memory-efficient attention |
| **DeepSpeed** | 0.16.9 | Distributed training optimization |
| **BitsAndBytes** | 0.49.0 | 4-bit/8-bit quantization |
| **Triton** | 3.4.0 | GPU compiler |

#### Data Processing

| Component | Version | Description |
|-----------|---------|-------------|
| **Datasets** | 3.6.0 | Hugging Face Datasets |
| **Tokenizers** | 0.21.1 | Fast tokenizers |
| **SentencePiece** | 0.2.1 | Subword tokenization |
| **TikToken** | 0.12.0 | OpenAI's tokenizer |

### 📁 Directory Structure

```
/
├── root/
│   ├── miniconda3/                    # Conda environment (system disk)
│   │   └── lib/python3.12/site-packages/
│   │       └── llamafactory/          # LLaMA-Factory installation
│   ├── autodl-tmp/                    # Data disk (fast I/O)
│   │   └── models/                    # Recommended model storage
│   └── autodl-fs/                     # Shared file storage
│       └── models/                    # Alternative model storage
```

### 🚀 Quick Start

#### 1. Launch WebUI

```bash
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=6006 llamafactory-cli webui
```

Then access the WebUI via **port 6006** in AutoDL's "Custom Service" panel.

#### 2. Download Models (with China Mirror)

```bash
# Set HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com

# Download Qwen2.5-7B-Instruct
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir /root/autodl-tmp/models/Qwen2.5-7B-Instruct

# Download Llama-3-8B-Instruct
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir /root/autodl-tmp/models/Llama-3-8B-Instruct
```

#### 3. Alternative: ModelScope Download

```bash
pip install modelscope

python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir='/root/autodl-tmp/models')
print(f'Model downloaded to: {model_dir}')
"
```

### ⚙️ Pre-configured Mirrors

| Service | Mirror URL |
|---------|------------|
| **pip** | https://mirrors.aliyun.com/pypi/simple/ |
| **pip (backup)** | https://pypi.tuna.tsinghua.edu.cn/simple/ |
| **HuggingFace** | https://hf-mirror.com |

### 🔧 CLI Commands

```bash
# Check version
llamafactory-cli version

# Launch WebUI
llamafactory-cli webui

# Start training (CLI mode)
llamafactory-cli train examples/train_lora/qwen2_lora_sft.yaml

# Chat with model
llamafactory-cli chat --model_name_or_path /root/autodl-tmp/models/Qwen2.5-7B-Instruct

# Export model
llamafactory-cli export --model_name_or_path /path/to/model --export_dir /path/to/export
```

### 📊 Supported GPUs

| GPU | VRAM | Architecture | Compute Capability |
|-----|------|--------------|-------------------|
| RTX 5090 | 32GB | Blackwell | SM 100 |
| RTX PRO6000 | 48GB | Blackwell | SM 100 |
| RTX 4090 | 24GB | Ada Lovelace | SM 89 |
| RTX 4080 | 16GB | Ada Lovelace | SM 89 |

### ⚠️ Important Notes

1. **Data Storage**: Store large models on `/root/autodl-tmp/` (data disk) for better I/O performance
2. **Persistent Storage**: Files in `/root/autodl-fs/` persist across instances
3. **Port Access**: Use port 6006 or 6008 for WebUI access via AutoDL's custom service
4. **Background Running**: Use `screen` or `nohup` for long-running training jobs

```bash
# Using screen
screen -S llama
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=6006 llamafactory-cli webui
# Press Ctrl+A, then D to detach

# Reconnect
screen -r llama
```

---

<a name="中文"></a>
## 🇨🇳 中文

### 简介

这是一个专为 **NVIDIA RTX 5090** 和 **RTX PRO6000** 显卡优化的 AutoDL 镜像，支持 **CUDA 12.8**，预装了 LLaMA-Factory 及所有必要依赖，包括 Flash Attention 2、DeepSpeed 和量化工具，开箱即用。

### ✨ 特性

- 🚀 **Blackwell 架构支持** - 完整支持 RTX 5090/PRO6000 (SM 100)
- ⚡ **Flash Attention 2** - 预装，最大化训练效率
- 🔧 **DeepSpeed 集成** - 支持分布式训练
- 📦 **量化工具** - BitsAndBytes 支持 4-bit/8-bit 训练
- 🌏 **国内镜像已配置** - 中国大陆高速下载

### 📦 环境配置

#### 系统环境

| 组件 | 版本 | 说明 |
|------|------|------|
| **操作系统** | Ubuntu 22.04 | 基础操作系统 |
| **Python** | 3.12 | Python 运行时 |
| **CUDA** | 12.8 | NVIDIA CUDA 工具包 |
| **cuDNN** | 9.x | 深度神经网络库 |

#### 核心依赖

| 组件 | 版本 | 说明 |
|------|------|------|
| **PyTorch** | 2.8.0+cu128 | 深度学习框架 |
| **LLaMA-Factory** | 0.9.3 | 微调框架 |
| **Transformers** | 4.52.4 | Hugging Face Transformers |
| **PEFT** | 0.15.2 | 参数高效微调 |
| **TRL** | 0.9.6 | 强化学习训练 |
| **Accelerate** | 1.7.0 | Hugging Face 加速库 |

#### 加速与优化组件

| 组件 | 版本 | 说明 |
|------|------|------|
| **Flash Attention** | 2.8.3 | 内存高效注意力机制 |
| **DeepSpeed** | 0.16.9 | 分布式训练优化 |
| **BitsAndBytes** | 0.49.0 | 4-bit/8-bit 量化 |
| **Triton** | 3.4.0 | GPU 编译器 |

#### 数据处理组件

| 组件 | 版本 | 说明 |
|------|------|------|
| **Datasets** | 3.6.0 | Hugging Face 数据集 |
| **Tokenizers** | 0.21.1 | 快速分词器 |
| **SentencePiece** | 0.2.1 | 子词分词 |
| **TikToken** | 0.12.0 | OpenAI 分词器 |

### 📁 目录结构

```
/
├── root/
│   ├── miniconda3/                    # Conda 环境（系统盘）
│   │   └── lib/python3.12/site-packages/
│   │       └── llamafactory/          # LLaMA-Factory 安装位置
│   ├── autodl-tmp/                    # 数据盘（高速 I/O）
│   │   └── models/                    # 推荐的模型存放位置
│   └── autodl-fs/                     # 共享文件存储
│       └── models/                    # 备选模型存放位置
```

### 🚀 快速开始

#### 1. 启动 WebUI

```bash
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=6006 llamafactory-cli webui
```

启动后，在 AutoDL 控制台点击「自定义服务」的 **6006 端口** 访问 WebUI。

#### 2. 下载模型（使用国内镜像）

```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 下载 Qwen2.5-7B-Instruct
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir /root/autodl-tmp/models/Qwen2.5-7B-Instruct

# 下载 Llama-3-8B-Instruct
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir /root/autodl-tmp/models/Llama-3-8B-Instruct
```

#### 3. 备选：使用 ModelScope 下载

```bash
pip install modelscope

python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir='/root/autodl-tmp/models')
print(f'模型已下载到: {model_dir}')
"
```

### ⚙️ 已配置的国内镜像

| 服务 | 镜像地址 |
|------|----------|
| **pip 主源** | https://mirrors.aliyun.com/pypi/simple/ |
| **pip 备用** | https://pypi.tuna.tsinghua.edu.cn/simple/ |
| **HuggingFace** | https://hf-mirror.com |

### 🔧 常用命令

```bash
# 查看版本
llamafactory-cli version

# 启动 WebUI
llamafactory-cli webui

# 命令行训练
llamafactory-cli train examples/train_lora/qwen2_lora_sft.yaml

# 模型对话
llamafactory-cli chat --model_name_or_path /root/autodl-tmp/models/Qwen2.5-7B-Instruct

# 导出模型
llamafactory-cli export --model_name_or_path /path/to/model --export_dir /path/to/export
```

### 📊 支持的显卡

| 显卡 | 显存 | 架构 | 算力 |
|------|------|------|------|
| RTX 5090 | 32GB | Blackwell | SM 100 |
| RTX PRO6000 | 48GB | Blackwell | SM 100 |
| RTX 4090 | 24GB | Ada Lovelace | SM 89 |
| RTX 4080 | 16GB | Ada Lovelace | SM 89 |

### ⚠️ 重要提示

1. **数据存储**：建议将大模型存放在 `/root/autodl-tmp/`（数据盘），I/O 性能更好
2. **持久存储**：`/root/autodl-fs/` 中的文件在实例间共享且持久保存
3. **端口访问**：通过 AutoDL「自定义服务」访问 6006 或 6008 端口
4. **后台运行**：长时间训练请使用 `screen` 或 `nohup`

```bash
# 使用 screen 后台运行
screen -S llama
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=6006 llamafactory-cli webui
# 按 Ctrl+A 然后按 D 分离会话

# 重新连接
screen -r llama
```

### 🔍 验证安装

```bash
python -c "
import torch
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA 可用: {torch.cuda.is_available()}')
print(f'✓ CUDA 版本: {torch.version.cuda}')
print(f'✓ GPU: {torch.cuda.get_device_name(0)}')

import flash_attn
print(f'✓ Flash Attention: {flash_attn.__version__}')

import deepspeed
print(f'✓ DeepSpeed: {deepspeed.__version__}')

import bitsandbytes
print(f'✓ BitsAndBytes: {bitsandbytes.__version__}')

import peft
print(f'✓ PEFT: {peft.__version__}')

import transformers
print(f'✓ Transformers: {transformers.__version__}')
"
```

---

## 📝 Changelog | 更新日志

### v1.0.0 (2025-12-19)
- 🎉 Initial release | 首次发布
- ✅ Full RTX 5090 support | 完整支持 RTX 5090
- ✅ CUDA 12.8 + PyTorch 2.8 | CUDA 12.8 + PyTorch 2.8
- ✅ Flash Attention 2.8.3 pre-installed | 预装 Flash Attention 2.8.3
- ✅ China mirrors pre-configured | 国内镜像已配置

---

## 📄 License | 许可证

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

本项目采用 Apache 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 Acknowledgments | 致谢

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - Unified Efficient Fine-Tuning of 100+ LLMs
- [AutoDL](https://www.autodl.com/) - GPU Cloud Platform
- [Hugging Face](https://huggingface.co/) - AI Community

---

<div align="center">

**Sponsored by 乐大师餐饮AI**

**Powered by 微信公众号：就是AI科技**

<img src="https://img.shields.io/badge/WeChat-就是AI科技-07C160?style=for-the-badge&logo=wechat&logoColor=white" alt="WeChat">

---

⭐ If this image helps you, please give it a star! | 如果这个镜像对你有帮助，请给个 Star！ ⭐

</div>
