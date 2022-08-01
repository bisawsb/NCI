import os
import argparse
import nltk
import pandas as pd
import time
import torch
import pytorch_lightning as pl

from main_metrics import recall, MRR100
from main_models import T5FineTuner, l1_query
from main_utils import set_seed, get_ckpt, dec_2d, numerical_decoder, decode_token
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from tqdm import tqdm
from transformers import T5Tokenizer

nltk.download('punkt')
print(torch.__version__)  # 1.10.0+cu113
print(pl.__version__)  # 1.4.9

logger = None
YOUR_API_KEY = '' # TODO: set your wandb key first

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

def train(args):
    model = T5FineTuner(args)
    save_mode = "min" if args.ckpt_monitor == 'train_loss' else "max"
    file_name_template = args.tag_info + '_{epoch}-{' + args.ckpt_monitor + ':.6f}'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename=file_name_template,
        save_on_train_epoch_end=args.ckpt_monitor == 'avg_train_loss',
        monitor=args.ckpt_monitor,
        mode=save_mode,
        save_top_k=1,
        every_n_val_epochs=args.check_val_every_n_epoch,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor()
    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        precision=16 if args.fp_16 else 32,
        amp_level=args.opt_level,
        resume_from_checkpoint=args.resume_from_checkpoint,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=True,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback],
        plugins=DDPPlugin(find_unused_parameters=False),
        accelerator=args.accelerator
    )
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)


def inference(args):
    model = T5FineTuner(args, train=False)
    ckpt_path, ckpt_epoch = get_ckpt(args)
    state_dict = torch.load(ckpt_path)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)
    num_samples = args.n_test if args.n_test >= 0 else None
    dataset = l1_query(args, tokenizer, num_samples=num_samples,  task='test')
    model.to("cuda")
    model.eval()
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False)
    inf_result_cache = []
    for batch in tqdm(loader):
        lm_labels = batch["target_ids"].numpy().copy()
        lm_labels[lm_labels[:, :] == model.tokenizer.pad_token_id] = -100
        if args.decode_embedding:
            if args.position:
                expand_scale = args.max_output_length if not args.hierarchic_decode else 1
                decode_vocab_size = args.output_vocab_size * expand_scale + 2
            else:
                decode_vocab_size = 12
        else:
            decode_vocab_size = None

        with torch.no_grad():
            outs = model.model.generate(
                batch["source_ids"].cuda(),
                attention_mask=batch["source_mask"].cuda(),
                use_cache=True,
                decoder_attention_mask=batch['target_mask'].cuda(),
                max_length=args.max_output_length,
                num_beams=args.num_return_sequences,
                length_penalty=args.length_penalty,
                num_return_sequences=args.num_return_sequences,
                early_stopping=False,
                decode_embedding=args.decode_embedding,
                decode_vocab_size=decode_vocab_size,
                decode_tree=model.root,
            )

        if args.decode_embedding == 1:
            dec = [numerical_decoder(args, ids, output=True) for ids in outs]
        elif args.decode_embedding == 2:
            dec = decode_token(args, outs.cpu().numpy())
        else:
            dec = [tokenizer.decode(ids) for ids in outs]

        texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
        dec = dec_2d(dec, args.num_return_sequences)
        for r in batch['rank']:
            gt = [s[:args.max_output_length - 2] for s in list(r[0])] if args.label_length_cutoff else list(r[0])
            ra = r[1]
            ra = [str(a.item()) for a in ra]

            for pred, g, text, ran in zip(dec, gt, texts, ra):
                pred = ','.join(pred)
                inf_result_cache.append([text, pred, g, int(ran)])

    res = pd.DataFrame(inf_result_cache, columns=["query", "pred", "gt", "rank"])
    res.sort_values(by=['query', 'rank'], ascending=True, inplace=True)
    res1 = res.loc[res['rank'] == 1]
    res1.to_csv(args.res1_save_path, mode='w', sep="\t", header=None, index=False)
    recall_value = recall(args)
    mrr_value = MRR100(args)
    return recall_value, mrr_value


def parsers_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--model_name_or_path', type=str, default="t5-")
    parser.add_argument('--tokenizer_name_or_path', type=str, default="t5-")
    parser.add_argument('--use_new_id', type=int, default=1, choices=[0, 1])
    parser.add_argument('--freeze_encoder', type=int, default=0, choices=[0, 1])
    parser.add_argument('--freeze_embeds', type=int, default=0, choices=[0, 1])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--num_train_epochs', type=int, default=500)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--n_val', type=int, default=-1)
    parser.add_argument('--n_train', type=int, default=-1)
    parser.add_argument('--n_test', type=int, default=-1)
    parser.add_argument('--early_stop_callback', type=int, default=0, choices=[0, 1])
    parser.add_argument('--fp_16', type=int, default=0, choices=[0, 1])
    parser.add_argument('--opt_level', type=str, default='O1')
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrain_encoder', type=int, default=1, choices=[0, 1])
    parser.add_argument('--limit_val_batches', type=float, default=1.0)
    parser.add_argument('--aug', type=int, default=0, choices=[0, 1])
    parser.add_argument('--accelerator', type=str, default="ddp")
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--num_cls', type=int, default=1000)
    parser.add_argument('--decode_embedding', type=int, default=2, choices=[0, 1, 2])
    parser.add_argument('--output_vocab_size', type=int, default=10)
    parser.add_argument('--hierarchic_decode', type=int, default=0, choices=[0, 1])
    parser.add_argument('--tie_word_embedding', type=int, default=0, choices=[0, 1])
    parser.add_argument('--tie_decode_embedding', type=int, default=1, choices=[0, 1])
    parser.add_argument('--out_file', type=str, default="res.tsv")
    parser.add_argument('--length_penalty', type=int, default=0.3)
    parser.add_argument('--query_generation', type=float, default=1)

    ## query process args
    parser.add_argument('--recall_num', type=list, default=[1, 5, 10], help='[1,5,10,20,50,100]')
    parser.add_argument('--random_gen', type=int, default=0, choices=[0, 1])
    parser.add_argument('--label_length_cutoff', type=int, default=0)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--val_check_interval', type=float, default=1.0)
    parser.add_argument('--test_set', type=str, default="dev")
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--max_input_length', type=int, default=20, help='If using prhase or keywords, set it to 1-5.')
    parser.add_argument('--inf_max_input_length', type=int, default=20)
    parser.add_argument('--max_output_length', type=int, default=10)
    parser.add_argument('--doc_length', type=int, default=64)
    # parser.add_argument('--old_data', type=int, default=1)
    # parser.add_argument('--contrastive_variant', type=str, default="", help='E_CL, ED_CL, doc_Reweight')
    parser.add_argument('--num_return_sequences', type=int, default=20, help='generated id num (include invalid)')
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--mode', type=str, default="train", choices=['train', 'eval'])
    parser.add_argument('--cluster', type=int, default=0, help='1--cluster, 0--server')
    parser.add_argument('--query_type', type=str, default='gtq_qg10',
                        help='gtq -- use ground turth query;'
                             'qg -- use qg; doc -- use doc token; '
                             'keyword, keyword_phrase, keyword_phrase_doc')
    parser.add_argument('--cat_keywords', type=int, default=1, help='if $cat_keywords > 1, '
                                        'then $cat_keywords keywords will be sampled and cat as a query')
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--decoder_learning_rate', type=float, default=1e-4)
    parser.add_argument('--certain_epoch', type=int, default=None)
    parser.add_argument('--given_ckpt', type=str, default='')
    parser.add_argument('--model_info', type=str, default='base', choices=['large', 'base', '3b', '11b'])
    parser.add_argument('--traindata_num', type=str, default='', help='If using part dataset eg.train10, set it to _10.')
    parser.add_argument('--msmarco', type=int, default=0, help='If using NQ dataset, set it to 0.')
    parser.add_argument('--nq_qgnum', type=str, default='10', help='nq_qgnum, set it 1/5/10.')
    parser.add_argument('--id_method', type=str, default='univ_NQ',
                    help='new_id (bert k-means, max_output_length = 10, label_length_cutoff = 1),'
                         'old_id (just id, max_output_length = 8, label_length_cutoff = 0), '
                         'processed_NQ (Chengmin k-means), max_output_length = 12, label_length_cutoff = 0,'
                         'univ_NQ, all ids.')
    parser.add_argument('--id_class', type=str, default='randid',
                        help="randid, randstrid, bert_512_k10_c100, bert_512_k10_c10, bert_64_k10_c10")
    parser.add_argument('--ckpt_monitor', type=str, default='recall1', choices=['recall1', 'avg_train_loss'])
    parser.add_argument('--Rdrop', type=float, default=0.01, help='default to 0, 0.2')
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--Rdrop_only_decoder', type=int, default=0,
                        help='1-RDrop only for decoder, 0-RDrop only for all model', choices=[0,1])
    parser.add_argument('--adaptor_decode', type=int, default=1, help='default to 0,1')
    parser.add_argument('--adaptor_layer_num', type=int, default=1)
    parser.add_argument('--msmarco_folder', type=str, default='', help='_10_qg')
    parser.add_argument('--position', type=int, default=1)
    parser_args = parser.parse_args()

    # args post process
    parser_args.tokenizer_name_or_path += parser_args.model_info
    parser_args.model_name_or_path += parser_args.model_info

    parser_args.gradient_accumulation_steps = max(int(8 / parser_args.n_gpu), 1)
    if parser_args.mode == 'train' and 'doc' in parser_args.query_type:
        assert parser_args.contrastive_variant == ''
        parser_args.max_input_length = parser_args.doc_length

    if parser_args.mode == 'train':
        # set to small val to prevent CUDA OOM
        parser_args.num_return_sequences = 10
        parser_args.eval_batch_size = 1

    if parser_args.model_info == 'base':
        parser_args.num_layers = 12
        parser_args.num_decoder_layers = 6
        parser_args.d_ff = 3072
        parser_args.d_model = 768
        parser_args.num_heads = 12
        parser_args.d_kv = 64
    elif parser_args.model_info == 'large':
        parser_args.num_layers = 24
        parser_args.num_decoder_layers = 12
        parser_args.d_ff = 4096
        parser_args.d_model = 1024
        parser_args.d_kv = 64
        parser_args.num_heads = 16
    elif parser_args.model_info == '3b':
        parser_args.num_layers = 24
        parser_args.num_decoder_layers = 12
        parser_args.d_ff = 16384
        parser_args.d_model = 1024
        parser_args.num_heads = 32
        parser_args.d_kv = 128
        parser_args.decoder_start_token_id = 0
        parser_args.eos_token_id = 1
    elif parser_args.model_info == '11b':
        parser_args.num_layers = 24
        parser_args.num_decoder_layers = 12
        parser_args.d_kv = 128
        parser_args.d_ff = 65536
        parser_args.d_model = 1024
        parser_args.num_heads = 128
    elif parser_args.model_info == 'small':
        parser_args.num_layers = 6
        parser_args.num_decoder_layers = 3
        parser_args.d_ff = 2048
        parser_args.d_model = 512
        parser_args.num_heads = 8
        parser_args.d_kv = 64

    if parser_args.id_method == 'univ_NQ':
        if 'rand' in parser_args.id_class:
            parser_args.max_output_length = 8
            parser_args.label_length_cutoff = 0
        elif parser_args.id_class == 'bert_512_k10_c100':
            if parser_args.msmarco:
                    parser_args.max_output_length = 12
            else:
                parser_args.max_output_length = 9
            parser_args.label_length_cutoff = 0
        elif parser_args.id_class == 'bert_512_k10_c10':
            parser_args.max_output_length = 11
            parser_args.label_length_cutoff = 0
        elif parser_args.id_class == 'bert_64_k10_c10':
            parser_args.max_output_length = 11
            parser_args.label_length_cutoff = 0
        elif parser_args.id_class == 'k10_c10':
            if parser_args.msmarco:
                parser_args.max_output_length = 15
            else:
                parser_args.max_output_length = 9
            parser_args.label_length_cutoff = 0
    else:
        print("Error id_method.")
        exit(1)

    return parser_args


if __name__ == "__main__":
    args = parsers_parser()
    set_seed(args.seed)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    if args.cluster:
        dir_path = '/amulet/projects/corpus'
        parent_path = '/amulet/projects'
    print(dir_path)
    print(parent_path)
    args.logs_dir = dir_path + '/logs/'
    args.data_dir = dir_path + '/corpusindex_dataset/'
    # this is model pkl save dir
    args.output_dir = dir_path + '/logs/'

    ###########################
    time_str = time.strftime("%Y%m%d-%H%M%S")
    # TODO: Note -- you can put important info into here, then it will appear to the name of saved ckpt
    important_info_list = [args.query_type, args.model_info, args.id_class,
                           args.id_method, args.test_set, args.ckpt_monitor,  'dem:',
                           str(args.decode_embedding), 'ada:', str(args.adaptor_decode),
                           'RDrop:', str(args.dropout_rate), str(args.Rdrop), str(args.Rdrop_only_decoder),
                            args.msmarco_folder]

    args.query_info = '_'.join(important_info_list)
    if YOUR_API_KEY != '':
        os.environ["WANDB_API_KEY"] = YOUR_API_KEY
        logger = WandbLogger(name='{}-{}'.format(time_str, args.query_info), project='l1-t5-nq')
    else:
        logger = TensorBoardLogger("logs/")
    ###########################

    args.tag_info = '{}_lre{}d{}'.format(args.query_info, str(float(args.learning_rate * 1e4)), str(float(args.decoder_learning_rate * 1e4)))
    args.res1_save_path = args.logs_dir + '{}_res1_recall{}_{}.tsv'.format(args.tag_info, args.num_return_sequences, time_str)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        inference(args)