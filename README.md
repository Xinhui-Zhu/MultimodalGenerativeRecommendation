# Multimodal Generative Recommendation

## Installation

1. Install packages via `pip3 install -r requirements.txt`. 

2. Prepare `Amazon Reviews` Datasets:
    1. Download `Amazon Reviews` subsets: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/, process them by `process.py`, and put into the dataset and information folder. 
    2. Please note that Interactions and Item Information should be put into two folders like:
        ```bash
        ├── dataset # Store Interactions
        │   ├── arts.csv
        │   ├── arts_coldrec_ids.pkl
        │   ├── prime_pantry.csv
        │   └── prime_pantry_coldrec_ids.pkl
        └── information # Store Item Information
            ├── arts.csv
            └── prime_pantry.csv
        ``` 
        Here dataset represents **data_path**, and infomation represents **text_path**.
3. Prepare pre-trained LLM models, such as [TinyLlama](https://github.com/jzhang38/TinyLlama), [Baichuan2](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base).

## Training

Set `master_addr`, `master_port`, `nproc_per_node`, `nnodes` and `node_rank` in environment variables for multinodes training.

All hyper-parameters (except model's config) can be found in code/REC/utils/argument_list.py and passed through CLI. More model's hyper-parameters are in `IDNet/*` or `HLLM/*`. 

To reproduce our experiments on Amazon Reviews you can run scripts in `reproduce` folder.

## Inference

You can evaluate models you trained by the almost same command but `val_only = True`, you can run scripts in `reproduce/eval` folder.
