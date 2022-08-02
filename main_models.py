import torch
import numpy as np
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
from main_utils import assert_all_frozen, load_data_msmarco_infer, load_data_msmarco, numerical_decoder, dec_2d, \
    encode_single_newid, decode_token, TreeBuilder
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    get_linear_schedule_with_warmup
)

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader




class l1_query(Dataset):
    def __init__(self, args, tokenizer, num_samples, print_text=False, task='train'):
        assert task in ['train', 'test']
        self.args = args
        input_length = args.max_input_length
        output_length = args.max_output_length * int(np.log10(args.output_vocab_size))
        inf_input_length = args.inf_max_input_length
        random_gen = args.random_gen
        aug = args.aug
        if task == 'train':
            self.dataset = load_data_msmarco(args)
        elif task == 'test':
            self.dataset = load_data_msmarco_infer(args)

        if num_samples:
            self.dataset = self.dataset[:num_samples]

        self.task = task
        self.input_length = input_length
        self.doc_length = self.args.doc_length
        self.inf_input_length = inf_input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text
        self.aug = aug
        self.random_gen = random_gen
        if random_gen:
            assert len(self.dataset[0]) >= 3
        self.random_min = 2
        self.random_max = 6
        self.vocabs = set(self.tokenizer.get_vocab().keys())
        for token in [self.tokenizer.eos_token, self.tokenizer.unk_token, self.tokenizer.sep_token,
                      self.tokenizer.pad_token, self.tokenizer.cls_token,
                      self.tokenizer.mask_token] + tokenizer.additional_special_tokens:
            if token is not None:
                self.vocabs.remove(token)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def clean_text(text):
        text = text.replace('\n', '')
        text = text.replace('``', '')
        text = text.replace('"', '')

        return text

    def convert_to_features(self, example_batch, length_constraint):
        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch))

        input_ = self.clean_text(example_batch)
        output_ = self.tokenizer.batch_encode_plus([input_], max_length=length_constraint,
                                                   padding='max_length', truncation=True, return_tensors="pt")

        return output_

    def __getitem__(self, index):
        inputs = self.dataset[index]
        query, target, rank = inputs[0], inputs[1], inputs[2]
        if self.args.label_length_cutoff:
            target = target[:self.args.max_output_length - 2]
        source = self.convert_to_features(query, self.input_length if self.task == 'train' else self.inf_input_length)
        source_ids = source["input_ids"].squeeze()
        src_mask = source["attention_mask"].squeeze()

        targets = self.convert_to_features(target, self.output_length)
        target_ids = targets["input_ids"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        lm_labels = torch.zeros(self.args.max_output_length, dtype=torch.long)

        if self.args.decode_embedding:
            target_id = self.tokenizer.decode(target_ids)
            target_id_int = []
            bits = int(np.log10(self.args.output_vocab_size))
            idx = 0
            for i in range(0, len(target_id), bits):
                if i + bits >= len(target_id):
                    c = target_id[i:]
                c = target_id[i:i + bits]
                if self.args.position:
                    temp = idx * self.args.output_vocab_size + int(c) + 2 \
                        if not self.args.hierarchic_decode else int(c) + 2
                else:
                    temp = int(c) + 2
                target_id_int.append(temp)
                idx += 1
            lm_labels[:len(target_id_int)] = torch.LongTensor(target_id_int)
            lm_labels[len(target_id_int)] = 1
            decoder_attention_mask = lm_labels.clone()
            decoder_attention_mask[decoder_attention_mask != 0] = 1
            target_ids = lm_labels
            target_mask = decoder_attention_mask

        return {"source_ids": source_ids,
                "source_mask": src_mask,
                "target_ids": target_ids,
                "target_mask": target_mask,
                "doc_ids":  torch.tensor([-1997], dtype=torch.int64),
                "doc_mask":  torch.tensor([-1997], dtype=torch.int64),
                "softmax_index": torch.tensor([-1997], dtype=torch.int64),
                "rank": rank}


class T5FineTuner(pl.LightningModule):
    def __init__(self, args, train=True):
        super(T5FineTuner, self).__init__()
        builder = TreeBuilder()
        if not args.msmarco:
            df_train = pd.read_csv(
                args.data_dir + 'NQ_dataset/' + args.id_method + '/nq_train_doc_newid.tsv',
                names=["query", "queryid", "bert_512_k10_c100"],
                encoding='utf-8', header=None, sep='\t',
                dtype={'query': str, 'queryid': str, 'bert_512_k10_c100': str}).loc[:,
                ["query", "queryid", args.id_class]]
            df_dev = pd.read_csv(
                args.data_dir + 'NQ_dataset/' + args.id_method + '/nq_dev_doc_newid.tsv',
                names=["query", "queryid", "bert_512_k10_c100"],
                encoding='utf-8', header=None, sep='\t',
                dtype={'query': str, 'queryid': str, 'bert_512_k10_c100': str}).loc[:,
                    ["query", "queryid", args.id_class]]
            df = pd.merge(df_train, df_dev, how='outer')
        else:
            df_train = pd.read_csv(
                args.data_dir + 'msmarco/univ_MS{}/msmarco_train_query2new_id_new.tsv'.format(args.msmarco_folder),
                encoding='utf-8', names=["query", "k10_c10", "k10_c100"], header=None, sep='\t',
                dtype={'query': str, 'k10_c10': str
                       }).loc[:, ["query", args.id_class]]
            df_dev = pd.read_csv(
                args.data_dir + 'msmarco/univ_MS{}/msmarco_dev_query2new_id_new.tsv'.format(args.msmarco_folder),
                encoding='utf-8', names=["query", "k10_c10", "k10_c100"], header=None, sep='\t',
                dtype={'query': str, 'k10_c10': str
                       }).loc[:, ["query", args.id_class]]
            df = pd.merge(df_train, df_dev, how='outer')

        for _, (_, newid) in df.iterrows():
            toks = encode_single_newid(args, newid)
            builder.add(toks)

        root = builder.build()
        self.root = root

        self.args = args
        self.save_hyperparameters(args)
        if args.decode_embedding:
            if self.args.position:
                expand_scale = args.max_output_length if not args.hierarchic_decode else 1
                self.decode_vocab_size = args.output_vocab_size * expand_scale + 2
            else:
                self.decode_vocab_size = 12
        else:
            self.decode_vocab_size = None

        t5_config = T5Config(
            num_layers=args.num_layers,
            num_decoder_layers=args.num_decoder_layers,
            d_ff=args.d_ff,
            d_model=args.d_model,
            num_heads=args.num_heads,
            decoder_start_token_id=0,
            output_past=True,
            d_kv=args.d_kv,
            dropout_rate=args.dropout_rate,
            decode_embedding=args.decode_embedding,
            hierarchic_decode=args.hierarchic_decode,
            decode_vocab_size=self.decode_vocab_size,
            tie_word_embeddings=args.tie_word_embedding,
            tie_decode_embedding=args.tie_decode_embedding,
            Rdrop=args.Rdrop,
            Rdrop_only_decoder=args.Rdrop_only_decoder,
            adaptor_decode=args.adaptor_decode,
            adaptor_layer_num=args.adaptor_layer_num,
        )
        model = T5ForConditionalGeneration(t5_config)
        if args.pretrain_encoder:
            pretrain_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
            pretrain_params = dict(pretrain_model.named_parameters())
            for name, param in model.named_parameters():
                if name.startswith(("shared.", "encoder.")):
                    with torch.no_grad():
                        param.copy_(pretrain_params[name])
        self.model = model
        self.tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)

        if self.args.freeze_embeds:
            self.freeze_embeds()
        if self.args.freeze_encoder:
            self.freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())

        n_observations_per_split = {
            "train": self.args.n_train,
            "validation": self.args.n_val,
            "test": self.args.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        if train:
            n_samples = self.n_obs['train']
            train_dataset = l1_query(self.args, self.tokenizer, n_samples)
            self.t_total = (
                    (len(train_dataset) // (self.args.train_batch_size * max(1, self.args.n_gpu)))
                    // self.args.gradient_accumulation_steps
                    * float(self.args.num_train_epochs)
            )

    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            self.freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                self.freeze_params(d.embed_positions)
                self.freeze_params(d.embed_tokens)
        except AttributeError:
            self.freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                self.freeze_params(d.embed_tokens)

    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))

    def is_logger(self):
        return self.trainer.global_rank <= 0

    def parse_score(self, result):
        return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
        if self.args.Rdrop > 0 and not self.args.Rdrop_only_decoder:
            input_ids = torch.cat([input_ids, input_ids.clone()], dim=0)
            attention_mask = torch.cat([attention_mask, attention_mask.clone()], dim=0)
            decoder_attention_mask = torch.cat([decoder_attention_mask, decoder_attention_mask.clone()], dim=0)
            lm_labels = torch.cat([lm_labels, lm_labels.clone()], dim=0)
            if decoder_input_ids is not None:
                decoder_input_ids = torch.cat([decoder_input_ids, decoder_input_ids.clone()], dim=0)

        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            lm_labels=lm_labels,
            return_dict=True,
        )

        return out

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self.forward(input_ids=batch["source_ids"], attention_mask=batch["source_mask"],
                               lm_labels=lm_labels, decoder_attention_mask=batch['target_mask'], )
        loss = outputs.loss

        if self.args.Rdrop > 0:
            orig_loss = outputs.orig_loss
            dist_loss = outputs.dist_loss
        else:
            orig_loss, dist_loss = 0, 0
        return loss, orig_loss, dist_loss

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)

    def training_step(self, batch, batch_idx):
        loss, orig_loss, kl_loss = self._step(batch)
        self.log("train_loss", loss)
        return {"loss": loss, "orig_loss": orig_loss, "kl_loss": kl_loss}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_train_loss", avg_train_loss)
        if self.args.Rdrop > 0:
            avg_orig_loss = torch.stack([x["orig_loss"] for x in outputs]).mean()
            avg_kl_loss = torch.stack([x["kl_loss"] for x in outputs]).mean()
            self.log("avg_train_orig_loss", avg_orig_loss)
            self.log("avg_train_kl_loss", avg_kl_loss)

    def validation_step(self, batch, batch_idx):
        # set to eval
        inf_result_cache = []
        lm_labels = batch["target_ids"].cpu().numpy().copy()
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        if self.args.decode_embedding:
            if self.args.position:
                expand_scale = self.args.max_output_length if not self.args.hierarchic_decode else 1
                decode_vocab_size = self.args.output_vocab_size * expand_scale + 2
            else:
                decode_vocab_size = 12
        else:
            decode_vocab_size = None

        if self.args.decode_embedding == 1:
            outs, scores = self.model.generate(
                batch["source_ids"].cuda(),
                attention_mask=batch["source_mask"].cuda(),
                use_cache=True,
                decoder_attention_mask=batch['target_mask'].cuda(),
                max_length=self.args.max_output_length,
                num_beams=self.args.num_return_sequences,
                length_penalty=self.args.length_penalty,
                num_return_sequences=self.args.num_return_sequences,
                early_stopping=False,
                decode_embedding=self.args.decode_embedding,
                decode_vocab_size=decode_vocab_size,
                output_scores=True
            )
            dec = [numerical_decoder(self.args, ids, output=True) for ids in outs]
        elif self.args.decode_embedding == 2:
            outs, scores = self.model.generate(
                batch["source_ids"].cuda(),
                attention_mask=batch["source_mask"].cuda(),
                use_cache=False,
                decoder_attention_mask=batch['target_mask'].cuda(),
                max_length=self.args.max_output_length,
                num_beams=self.args.num_return_sequences,
                length_penalty=self.args.length_penalty,
                num_return_sequences=self.args.num_return_sequences,
                early_stopping=False,
                decode_embedding=self.args.decode_embedding,
                decode_vocab_size=decode_vocab_size,
                decode_tree=self.root,
                output_scores=True
            )
            dec = decode_token(self.args, outs.cpu().numpy())
        else:
            outs, scores = self.model.generate(
                batch["source_ids"].cuda(),
                attention_mask=batch["source_mask"].cuda(),
                use_cache=True,
                decoder_attention_mask=batch['target_mask'].cuda(),
                max_length=self.args.max_output_length,
                num_beams=self.args.num_return_sequences,
                length_penalty=self.args.length_penalty,
                num_return_sequences=self.args.num_return_sequences,
                early_stopping=False,
                decode_embedding=self.args.decode_embedding,
                decode_vocab_size=decode_vocab_size,
                decode_tree=self.root,
                output_scores=True
            )
            dec = [self.tokenizer.decode(ids) for ids in outs]

        texts = [self.tokenizer.decode(ids) for ids in batch['source_ids']]

        dec = dec_2d(dec, self.args.num_return_sequences)
        for r in batch['rank']:
            gt = [s[:self.args.max_output_length - 2] for s in list(r[0])] if self.args.label_length_cutoff else list(r[0])
            ra = r[1]
            ra = [str(a.item()) for a in ra]

            for pred, g, text, ran in zip(dec, gt, texts, ra):
                pred = ','.join(pred)
                inf_result_cache.append([text, pred, g, int(ran)])
        return {"inf_result_batch": inf_result_cache, 'inf_result_batch_prob': scores}

    def validation_epoch_end(self, outputs):
        inf_result_cache = [item for sublist in outputs for item in sublist['inf_result_batch']]

        res = pd.DataFrame(inf_result_cache, columns=["query", "pred", "gt", "rank"])
        res.sort_values(by=['query', 'rank'], ascending=True, inplace=True)
        res1 = res.loc[res['rank'] == 1]
        res1 = res1.values.tolist()

        q_gt, q_pred = {}, {}
        prev_q = ""
        for [query, pred, gt, _] in res1:
            if query != prev_q:
                q_pred[query] = pred.split(",")
                q_pred[query] = q_pred[query][:1]
                q_pred[query] = set(q_pred[query])
                prev_q = query
            if query in q_gt:
                if len(q_gt[query]) <= 100:
                    q_gt[query].add(gt)
            else:
                q_gt[query] = set()
                q_gt[query].add(gt)

        total = 0
        for q in q_pred:
            right = 0
            wrong = 0
            for p in q_gt[q]:
                if p in q_pred[q]:
                    right += 1
                else:
                    wrong += 1
            recall = right / (right + wrong)
            total += recall
        recall_avg = total / len(q_pred)
        print("recall@1:{}".format(recall_avg))
        self.log("recall1", recall_avg)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if
                           (not any(nd in n for nd in no_decay)) and (n.startswith(("shared.", "encoder.")))],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           (not any(nd in n for nd in no_decay)) and (not n.startswith(("shared.", "encoder.")))],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.decoder_learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           (any(nd in n for nd in no_decay)) and (n.startswith(("shared.", "encoder.")))],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           (any(nd in n for nd in no_decay)) and (not n.startswith(("shared.", "encoder.")))],
                "weight_decay": 0.0,
                "lr": self.args.decoder_learning_rate,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.t_total
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        n_samples = self.n_obs['train']
        train_dataset = l1_query(self.args, self.tokenizer, n_samples)
        sampler = DistributedSampler(train_dataset)
        dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=self.args.train_batch_size,
                                drop_last=True, shuffle=False, num_workers=4)
        return dataloader

    def val_dataloader(self):
        n_samples = self.n_obs['validation']
        val_dataset = l1_query(self.args, self.tokenizer, n_samples, task='test')
        sampler = DistributedSampler(val_dataset)
        dataloader = DataLoader(val_dataset, sampler=sampler, batch_size=self.args.train_batch_size,
                                drop_last=True, shuffle=False, num_workers=4)
        return dataloader