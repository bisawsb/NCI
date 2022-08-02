#!/usr/bin/env bash
export PYTHONPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python main.py --cluster 0 --decode_embedding 2 --n_gpu 8 --mode train --query_type gtq_qg10 \
--train_batch_size 64 --model_info base --id_class bert_512_k10_c100 --adaptor_decode 1 \
--dropout_rate 0.1  --Rdrop 0.01 --msmarco 0