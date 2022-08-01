# A Neural Corpus Indexer for Document Retrieval (NCI)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python3-1f425f.svg?color=purple)](https://www.python.org/)

## What is NCI?

**NCI** is an end-to-end, sequence-to-sequence differentiable **document retrieval model** which retrieve relevant document identifiers directly for specific queries. In our evaluation on [Google NQ dataset](https://ai.google.com/research/NaturalQuestions), NCI outperforms all baselines and model-based indexers:

| Model | Recall@1 | Recall@10 | Recall@100 | MRR@100 |
|:-------:|:--------:|:-----:|:-----:|:-------:|
| **NCI (Ours)** | **88.72** | **95.84** | **97.43** | **91.59** |
| DSI (T5-Base) | 27.40 | 56.60 | -- | -- |
| SEAL (BART-Base) | 26.55 | 53.61 | 72.67 | 35.64 |
| ANCE (MaxP) | 52.63 | 80.38 | 91.31 | 62.84 |
| BM25 + DocT5Query | 58.39 | 75.76 | 89.51 | 64.53 |

For more information, checkout our publications:
*[removed for blind peer review]*

Cite as below if you find this repository is helpful to your project:

```
[removed for blind peer review]
```

## Environemnt
[1] Install [Anaconda](https://www.anaconda.com/download).

[2] Clone repository:

```bash
git clone https://github.com/anonymousML36061/NCI.git
cd NCI
```

[3] Create conda environment:

```bash
conda env create -f environment.yml
source activate NCI
```

## Data Process

Currently NCI is evaluated on [Google NQ dataset](https://ai.google.com/research/NaturalQuestions). Please download it before re-training and evalution.


### Semantic Identifier

NCI uses content-based document identifiers: A pre-trained BERT is used to generate document embeddings, and then documents are clustered using hierarchical K-means and semantic identifiers are assigned to each document.

Please find more details in [NQ_dataset_Process.ipynb](./NQ_dataset_Process.ipynb).



### Query Generation

In our study, Query Generation can significantly improve retrieve performance, especially for long-tail queries.

NCI uses [docTTTTTquery](https://github.com/castorini/docTTTTTquery) checkpoint to generate synthetic queries. Please refer to docTTTTTquery documentation and find more details in [NQ_dataset_Process.ipynb](./NQ_dataset_Process.ipynb).

<!---
### Data Process Pipeline

1. Confirm NQ dataset's setting. (Need to complement)
2. (Run k-means clustering code, and indicate the location of generated new document id.)
3. (Use QG script, and indicate the location of processed QG data.)
4. Run [NQ_dataset_Process.ipynb](./NQ_dataset_Process.ipynb). Conbine the previous data together into a format which can be used directly by training script, and put the processed train/dev data under `./corpusindex_datast` folder.
-->

## Training

Once the data pre-processing is complete, you can launch training by [train.sh](./train.sh).


## Evaluation
Please use [eval.sh](./eval.sh) along with [our checkpoint](https://drive.google.com/file/d/1Hh_TQNfvhKQzE0qFknK65z1_TSH6RuPK/view?usp=sharing) or your own checkpoint to evaluate model performance.


## Contact

Contact Person: *[removed for blind peer review]*

Please send us e-mail if any question. Welcome to any collaboration!

## Acknowledgement

We learned a lot and borrowed some code from the following projects when building NCI.
- [Transformers](https://github.com/huggingface/transformers)
- [docTTTTTquery](https://github.com/castorini/docTTTTTquery) 
