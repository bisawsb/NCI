#!/usr/bin/env bash

INFER_CKPT='nq_model.ckpt'
BEAM_SIZE=100

python main.py --cluster 0 --decode_embedding 2 --n_gpu 1 --mode eval --infer_ckpt $INFER_CKPT --num_return_sequences $BEAM_SIZE --query_type gtq_qg --train_batch_size 64 --model_info base --id_class bert_512_k10_c100 --msmarco 0 --eval_batch_size 16 --adaptor_decode 1