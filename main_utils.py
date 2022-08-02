import random
from typing import List

import torch
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



class Node(object):
    def __init__(self, token_id) -> None:
        self.token_id = token_id
        self.children = {}

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.token_id) + "\n"
        for child in self.children.values():
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return '<tree node representation>'


class TreeBuilder(object):
    def __init__(self) -> None:
        self.root = Node(0)

    def build(self) -> Node:
        return self.root

    def add(self, seq) -> None:
        '''
        seq is List[Int] representing id, without leading pad_token(hardcoded 0), with trailing eos_token(hardcoded 1) and (possible) pad_token, every int is token index
        e.g: [ 9, 14, 27, 38, 47, 58, 62,  1,  0,  0,  0,  0,  0,  0,  0]
        '''
        cur = self.root
        for tok in seq:
            if tok == 0:  # reach pad_token
                return
            if tok not in cur.children:
                cur.children[tok] = Node(tok)
            cur = cur.children[tok]


def encode_single_newid(args, seq):
    '''
    Param:
        seq: doc_id string to be encoded, like "23456"
    Return:
        List[Int]: encoded tokens
    '''
    target_id_int = []
    for i, c in enumerate(seq):
        if args.position:
            cur_token = i * 10 + int(c) + 2  # hardcoded vocab_size = 10
        else:
            cur_token = int(c) + 2
        target_id_int.append(cur_token)
    return target_id_int + [1]  # append eos_token


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def filter_seqs(seqs, decode_tree):
    '''
    Param:
        seqs: 2d numpy array, [batch_size, token_length]
        decode tree: Tree node
    Return:
        filtered sequences guaranteed in decode tree, 2d numpy array
    '''
    filter_idx = []
    seqs_list = seqs.tolist()
    for idx, seq in enumerate(seqs_list):
        # check one and only one eos_token in seq
        if seq.count(1) != 1:
            continue

        eos_idx = seq.index(1)
        # all taken after eos are pad_token
        if sum(seq[eos_idx + 1:]) != 0:
            continue

        # ignore leading pad_token
        seq = seq[1:eos_idx + 1]
        cur = decode_tree
        for tok in seq:
            if tok not in cur.children:
                break
            else:
                cur = cur.children[tok]
        else:
            if len(cur.children) == 0:  # reach leaf node once decoded finished
                filter_idx.append(idx)
    return np.take(seqs, filter_idx, axis=0)


def decode_token(args, seqs):
    '''
    Param:
        seqs: 2d ndarray to be decoded
    Return:
        doc_id string, List[str]
    '''
    result = []
    # assert np.all(np.count_nonzero(seqs == 1, axis=1) == 1), "zero or multiple eos token found"
    for seq in seqs:
        # print(seq)
        eos_idx = seq.tolist().index(1)
        seq = seq[1: eos_idx]
        if args.position:
            offset = np.arange(len(seq)) * 10 + 2  # hardcoded vocab_size = 10
        else:
            offset = 2
        res = seq - offset
        assert np.all(res >= 0)
        result.append(''.join(str(c) for c in res))
    return result


def get_ckpt(args):
    certain_epoch, given_ckpt, ckpt_saved_folder = args.certain_epoch, args.given_ckpt, args.logs_dir
    ckpt_files = [f for f in listdir(ckpt_saved_folder) if isfile(join(ckpt_saved_folder, f))]
    assert len(ckpt_files) >= 1
    desired_ckpt_name = ''
    desired_ckpt_epoch = 0
    if given_ckpt is not None:
        desired_ckpt_epoch = int(given_ckpt.split('=')[1].split('-')[0])
        desired_ckpt_name = ckpt_saved_folder + given_ckpt
    else:
        for ckpt_name in ckpt_files:
            if ckpt_name[-4:] != 'ckpt':
                continue
            if ckpt_name.split('_epoch')[0] != args.tag_info:
                continue

            try:
                ckpt_epoch = int(ckpt_name.split('epoch=')[1].split('-')[0])
            except:
                continue
            if certain_epoch is not None:
                if certain_epoch == ckpt_epoch:
                    desired_ckpt_epoch, desired_ckpt_name = ckpt_epoch, ckpt_name
            else:
                if ckpt_epoch > desired_ckpt_epoch:
                    desired_ckpt_epoch, desired_ckpt_name = ckpt_epoch, ckpt_name
    print('=' * 20)
    print('Loading: ' + desired_ckpt_name)
    print('=' * 20)
    assert desired_ckpt_name in ckpt_files
    return ckpt_saved_folder + desired_ckpt_name, desired_ckpt_epoch


def grad_status(model):
    return (par.requires_grad for par in model.parameters())


def lmap(f, x):
    """list(map(f, x))"""
    return list(map(f, x))


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad / npars:.1%} of {npars} weights require grad"


def dec_2d(dec, size):
    res = []
    i = 0
    while i < len(dec):
        res.append(dec[i: i + size])
        i = i + size
    return res


###### decoder helper
def numerical_decoder(args, cuda_ids, output):
    np_ids = cuda_ids.cpu().numpy()
    begin_and_end_token = np.where(np_ids == 1)

    if output:
        if len(begin_and_end_token) != 1 or begin_and_end_token[0].size < 1:
            print("Invalid Case")
            return "0"
        if args.hierarchic_decode:
            np_ids = np_ids[1:begin_and_end_token[0][0]] - 2
        else:
            np_ids = (np_ids[1:begin_and_end_token[0][0]] - 2) % args.output_vocab_size
    else:
        if args.hierarchic_decode:
            np_ids = np_ids[:begin_and_end_token[0][0]] - 2
        else:
            np_ids = (np_ids[:begin_and_end_token[0][0]] - 2) % args.output_vocab_size

    bits = int(np.log10(args.output_vocab_size))
    num_list = list(map(str, np_ids))
    str_ids = ''.join([c.zfill(bits) if (i != len(num_list) - 1) else c for i, c in enumerate(num_list)])
    return str_ids


def load_data_msmarco(args):
    def process_func1(index, query, newid, docid, rank=1):
        docid = ''.join([c for c in str(newid)]) if args.use_new_id else ''.join([c for c in str(docid)])
        softmax_index = int(str(newid)[:3]) if args.use_new_id else index
        return query, docid, rank, softmax_index

    def process_func2(index, query, newid, rank=1):
        docid = ''.join([c for c in str(newid)])
        softmax_index = int(str(newid)[:3])  if args.use_new_id else index
        return query, docid, rank, softmax_index


    if args.msmarco:
        df = pd.read_csv(
            args.data_dir + 'msmarco/univ_MS{}/msmarco_train_query2new_id_new.tsv'.format(args.msmarco_folder),
            encoding='utf-8', names=["query", "k10_c10", "k10_c100"],
            header=None, sep='\t', dtype={'query': str, 'k10_c10': str}).loc[:, ["query", args.id_class]]
        df[args.id_class] = df[args.id_class].map(lambda x: eval(x))
        df1 = df.explode(args.id_class)
        df = df1.drop_duplicates()
        assert not df.isnull().values.any()
        result = tuple(
            process_func2(index, *row) for index, row in
            enumerate(zip(df["query"], df[args.id_class]))
        )
    else:
        df = pd.read_csv(
            args.data_dir + 'NQ_dataset/' + args.id_method + '/nq_train_doc_newid.tsv',
            names=["query", "queryid", "bert_512_k10_c100"],
            encoding='utf-8', header=None, sep='\t',
            dtype={'query': str, 'queryid': str, 'bert_512_k10_c100': str}).loc[:,
             ["query", "queryid", args.id_class]]
        assert not df.isnull().values.any()
        result = tuple(
            process_func1(index, *row) for index, row in
            enumerate(zip(df["query"], df[args.id_class], df["queryid"]))
        )

    if 'qg' in args.query_type:
        # part 1
        if args.msmarco:
            gq_df1 = pd.read_csv(
                args.data_dir + 'msmarco/univ_MS{}/all_qg_10.tsv'.format(args.msmarco_folder),
                encoding='utf-8', names=["query", "k10_c10", "k10_c100"],
                header=None, sep='\t', dtype={'query': str, 'k10_c10': str
                                              }).loc[:, ["query", args.id_class]]
            gq_df1 = gq_df1.dropna(axis=0)
            result_add1 = tuple(
                process_func2(index, *row) for index, row in
                enumerate(zip(gq_df1["query"], gq_df1[args.id_class])))
        else:
            gq_df1 = pd.read_csv(
                args.data_dir + 'NQ_dataset/' + args.id_method + '/NQ_doc_qg.tsv',
                names=["query", "queryid", "bert_512_k10_c100"],
                encoding='utf-8', header=None, sep='\t',
                dtype={'query': str, 'queryid': str, 'bert_512_k10_c100': str}).loc[:,
                     ["query", "queryid", args.id_class]]
            gq_df1 = gq_df1.dropna(axis=0)
            result_add1 = tuple(
                process_func1(index, *row) for index, row in
                enumerate(zip(gq_df1["query"], gq_df1[args.id_class], gq_df1["queryid"])))
            filiter_none1 = list(result_add1)
            result_add1 = tuple(list(filter(None, filiter_none1)))

        result = result + result_add1

    return result



def load_data_msmarco_infer(args):
    df = None
    if args.test_set == 'dev':
        if args.msmarco:
            df = pd.read_csv(
                args.data_dir + 'msmarco/univ_MS{}/msmarco_dev_query2new_id_new.tsv'.format(args.msmarco_folder),
                encoding='utf-8', names=["query", "k10_c10", "k10_c100"],
                header=None, sep='\t', dtype={'query': str, 'k10_c10': str}).loc[:, ["query", args.id_class]]
        else:
            df = pd.read_csv(
                args.data_dir + 'NQ_dataset/' + args.id_method + '/nq_dev_doc_newid.tsv',
                 names=["query", "queryid", "bert_512_k10_c100"],
                encoding='utf-8', header=None, sep='\t',
                dtype={'query': str, 'queryid': str, 'bert_512_k10_c100': str}).loc[:,
                 ["query", "queryid", args.id_class]]

    assert not df.isnull().values.any()

    result = []
    softmax_index = -1
    for index, row in df.iterrows():
        query = row["query"]
        if not args.use_new_id:
            rank1 = ''.join([c for c in str(row["queryid"])])
        else:
            if not args.msmarco:
                rank1 = ''.join([c for c in str(row[args.id_class])])
            else:
                rank1 =  ''.join(
                    [c for c in str(eval(row[args.id_class])[0])])
        list_sum = []
        if not args.msmarco:
            docid = ''.join([c for c in str(row[args.id_class])]) if (args.use_new_id ) else ''.join(
                [c for c in str(row["queryid"])])
            if args.use_new_id:
                softmax_index = int(str(row[args.id_class])[:3])
            else:
                softmax_index += 1
        else:
            docid = ''.join([c for c in str(eval(row[args.id_class])[0])])
            if args.use_new_id:
                softmax_index = int(str(eval(row[args.id_class])[0])[:3])
            else:
                softmax_index += 1
        rank = 1

        list_sum.append((docid, rank))
        result.append((query, rank1, list_sum, softmax_index))

    return tuple(result)