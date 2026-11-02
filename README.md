<div align="center">

# TRIO: Task Refinement with Individual Optimization for Retrieval-Augmented Generation


</div>


## 🌟 Introduction

This repository is the official implementation of **EMNLP 2026 findins paper** TRIO: Task Refinement with Individual Optimization for Retrieval-Augmented Generation, a novel framework designed to improve Retrieval-Augmented Generation (RAG) systems.

---

## 🔧 Installation

Clone this repository, then create a conda environment and install the required packages.

```bash
# Clone repository
git clone https://github.com/yingtaoren/TRIO.git
cd TRIO

# Create conda environment
conda create -n trio python=3.10
conda activate trio

# Install packages
pip install -r requirements.txt
```

> 💡 **Note:** If you encounter any issues when installing the Python packages, we recommend following the official installation instructions provided by [FlashRAG#Installation](https://github.com/RUC-NLPIR/FlashRAG/tree/main#wrench-installation).

### Additional Dependencies

For faiss installation, use conda:

```bash
# CPU-only version
conda install -c pytorch faiss-cpu=1.8.0

# GPU(+CPU) version
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

---

## 📊 Dataset

The datasets used in this project follow the same format as those pre-processed by [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG). All datasets are available at [🤗 Huggingface datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets).

### Supported Datasets

| Type | Dataset 
|------|---------
| Multi-hop QA | HotpotQA 
| Multi-hop QA | 2WikiMultiHopQA 
| Multi-hop QA | MuSiQue 
| Multi-hop QA | Bamboogle 

### Dataset Setup

After downloading the datasets from FlashRAG, create a `/dataset` folder in the project directory and place the downloaded data inside:

```bash
TRIO
├── config
├── dataset
│   ├── 2wiki
│   ├── bamboogle
│   ├── HotpotQA
│   ├── musique
├── model
├── scripts
├── main.py
├── README.md
└── requirements.txt
```

> 💡 **Note:** If you wish to use a custom dataset path, modify the `data_dir` field in `config/base_config.yaml`.

---

## 📚 Document Corpus & Index

We use the `wiki18_100w` dataset as the document corpus. Both the document corpus and the index can be downloaded from:

- [ModelScope Dataset](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/files) (look in `retrieval_corpus` folder)


> 💡 **Note:** If you wish to use a custom corpus path, modify the `index_path` and `corpus_path` fields in `config/base_config.yaml`.

---

## 🤖 Models

### Retriever Model

We use `e5-base-v2` as the default retriever. Download from: [🤗 intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)

### Generator Models

This project supports all LLMs compatible with HuggingFace and vLLM. Recommended models:

| Model | Link |
|-------|------|
| Llama-3.1-8B-Instruct | [🔗 Link](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
| Qwen3-4B-Instruct | [🔗 Link](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) 

We also provided our trained moedls [here](https://huggingface.co/An998/TRIO).



Please specify the path to your downloaded model using the `model2path` field in `config/base_config.yaml`:

```yaml
model2path:
  e5: "/path/to/e5-base-v2"
  llama3-8B-instruct: "/path/to/Llama-3.1-8B-Instruct"
  # Add your custom models here
```

---

## 🏋️ Training

### Training Data Collection

For training data collection, please refer to the [DRAG repository](https://github.com/Huenao/Debate-Augmented-RAG) which provides the data collection pipeline.

### Supervised Fine-tuning (SFT)

For the judge agent training, run the SFT training script:

```bash
bash sft_train.sh
```


### Reinforcement Learning (RL)

For query-extension and answer agent training, we use [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). Run the training script:

```bash
bash my_train.sh
```


## 📈 Evaluation

### Running Evaluation

Evaluate TRIO on the supported datasets:

```bash
python main.py --method_name "TRIO" \
               --gpu_id "0" \
               --dataset_name "HotpotQA" \
               --generator_model "llama3-8B-instruct-TRIO" \
               --max_query_debate_rounds 3
```

---

## Citation

If you find our work useful, please consider citing our paper:

Paper: [TRIO: Task Refinement with Individual Optimization for Retrieval-Augmented Generation
]([https://dl.acm.org/doi/abs/10.1145/3805712.3809904](https://aclanthology.org/2026.findings-emnlp.937/))

```bibtex
@inproceedings{ren-etal-2026-trio,
    title = "{TRIO}: Task Refinement with Individual Optimization for Retrieval-Augmented Generation",
    author = "Ren, Yingtao  and
      Luo, Xiao  and
      Fu, Yiwei  and
      Zhou, Jinzhao  and
      Chang, Yu-Cheng  and
      Lin, Chin-teng",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2026",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-emnlp.937/",
    doi = "10.18653/2026.findings-emnlp.937",
    pages = "19765--19782",
    ISBN = "979-8-89176-462-0"
}
```

---


## ✨ Acknowledgments

We gratefully acknowledge the following projects:

- [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG): A Python toolkit for the reproduction and development of Retrieval Augmented Generation (RAG) research.
- [DRAG](https://github.com/Huenao/Debate-Augmented-RAG): Debate-Augmented RAG framework that inspired our work.
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF): An Easy-to-use, Scalable and High-performance RLHF Framework.

---

