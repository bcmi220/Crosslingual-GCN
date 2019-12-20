# -*- coding: utf-8 -*-
#
# extract_features.py - Pre-trained embeddings from language models
#
# Copyright 2018 The Google AI Language Team Authors.
# Copyright 2018 The HuggingFace Inc. team.
# Written in 2019 by Anonymous Author
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
#
# To the extent possible under law, the author(s) have dedicated all copyright
# and related and neighboring rights to this software to the public domain
# worldwide. This software is distributed without any warranty. You should have
# received a copy of the CC0 Public Domain Dedication along with this software.
# If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
#

import argparse
import collections
import logging
import json
import re
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

# from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.modeling import BertModel
from transformers import BertTokenizer, BertModel
from torchnlp.word_to_vector import FastText

# logging.basicConfig(
#     format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#     datefmt='%m/%d/%Y %H:%M:%S',
#     level=logging.INFO)
# logger = logging.get# logger(__name__)


class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask,
                 input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def add_args(parser):
    ## Required parameters
    parser.add_argument("--input-file",
                        default=None,
                        metavar="FILE",
                        nargs='+',
                        required=True)
    parser.add_argument("--output-file", default=None, type=str, required=True)
    parser.add_argument("--model",
                        default="",
                        type=str,
                        help="""Bert pre-trained model selected
                                in the list: bert-base-uncased, bert-large-uncased,
                                bert-base-cased, bert-base-multilingual,
                                bert-base-chinese.""")

    ## Other parameters
    parser.add_argument("--lowercase",
                        action='store_true',
                        help="""Set this flag if you are using an
                                uncased model.""")
    parser.add_argument('--lang',
                        type=str,
                        default="en",
                        metavar='LG',
                        help="""fastText language vectors.""")
    parser.add_argument("--layer", default="-2", type=int)
    parser.add_argument("--max-tokens",
                        default=32,
                        type=int,
                        help="""The maximum total input sequence
                                length after WordPiece tokenization.
                                Sequences longer than this will be truncated, and
                                sequences shorter than this will be padded.""")
    parser.add_argument("--batch-size",
                        default=32,
                        type=int,
                        help="Batch size for predictions.")
    parser.add_argument("--local-rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--pool',
                        type=str,
                        default="mean",
                        choices=['sum', 'mean', 'max', 'first', 'last'],
                        metavar='POOL',
                        help="""Pooling function.""")

    return parser


def tokenize(input_file, output_file, model, lowercase=False):
    if "bert" in model:
        tokenizer = BertTokenizer.from_pretrained(model,
                                                  do_lower_case=lowercase)
        tokenize_func = tokenizer.tokenize
    elif "xlmr" in model:
        xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large.v0')
        xlmr.eval()
        tokenize_func = xlmr.bpe.sp.EncodeAsPieces

    with open(input_file, 'r', encoding='utf-8') as reader:
        with open(output_file, 'w', encoding='utf-8') as writer:
            for line in reader:
                writer.write(' '.join(tokenize_func(line.rstrip())) + '\n')


def convert_examples_to_features(examples,
                                 seq_length,
                                 tokenizer,
                                 tokenized=True,
                                 lowercase=False):
    """Loads a data file into a list of `InputFeature`s."""

    features = []

    for example in examples:
        tokens_a = example.text_a.split() if tokenized else tokenizer.tokenize(
            example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = example.text_b.split(
            ) if tokenized else tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        # if ex_index < 5:
        # logger.info("*** Example ***")
        # logger.info("unique_id: %s" % (example.unique_id))
        # logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        # logger.info("input_ids: %s" % " ".join([str(x)
        #                                         for x in input_ids]))
        # logger.info("input_mask: %s" %
        #             " ".join([str(x) for x in input_mask]))
        # logger.info("input_type_ids: %s" %
        #             " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(unique_id=example.unique_id,
                          tokens=tokens,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          input_type_ids=input_type_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a,
                             text_b=text_b))
            unique_id += 1
    return examples


def write_features_from_examples(examples,
                                 output_file,
                                 bert_model,
                                 layer,
                                 device,
                                 batch_size,
                                 max_tokens,
                                 tokenized=False,
                                 local_rank=-1,
                                 n_gpu=0,
                                 lowercase=False,
                                 pool="cls"):

    tokenizer = BertTokenizer.from_pretrained(bert_model,
                                              do_lower_case=lowercase)
    features = convert_examples_to_features(examples=examples,
                                            seq_length=max_tokens,
                                            tokenizer=tokenizer,
                                            tokenized=tokenized)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model = BertModel.from_pretrained(bert_model, output_hidden_states=True)
    model.to(device)

    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    all_input_ids = torch.tensor([f.input_ids for f in features],
                                 dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features],
                                  dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    layers = [layer]

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    if local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data,
                                 sampler=eval_sampler,
                                 batch_size=batch_size)

    pool_func = lambda x: np.array(x[1:-1]).sum(axis=0).tolist()
    if pool == 'max':
        pool_func = lambda x: np.array(x[1:-1]).max(axis=0).tolist()
    elif pool == 'mean':
        pool_func = lambda x: np.array(x[1:-1]).mean(axis=0).tolist()
    elif pool == 'first':
        pool_func = lambda x: x[0]
    elif pool == 'last':
        pool_func = lambda x: x[-1]

    model.eval()
    with open(output_file, "w", encoding='utf-8') as writer:
        writer.write("{:d} {:d}\n".format(
            len(examples), model.embeddings.word_embeddings.weight.shape[1]))

        for input_ids, input_mask, example_indices in tqdm(eval_dataloader):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)

            _, _, all_encoder_layers = model(input_ids,
                                             token_type_ids=None,
                                             attention_mask=input_mask)
            # all_encoder_layers = all_encoder_layers

            for b, example_index in enumerate(example_indices):
                feature = features[example_index.item()]
                unique_id = int(feature.unique_id)
                # feature = unique_id_to_feature[unique_id]
                output_json = collections.OrderedDict()
                output_json["linex_index"] = unique_id
                all_out_features = []

                for (i, token) in enumerate(feature.tokens):
                    all_layers = []
                    for (j, layer_index) in enumerate(layers):
                        # layer_output = all_encoder_layers[j][b][i]
                        # layer_output = layer_output.detach().cpu().numpy()
                        # all_out_features.append(layer_output)
                        layer_output = all_encoder_layers[int(
                            layer_index)].detach().cpu().numpy()
                        values = [
                            round(x.item(), 6) for x in layer_output[b, i]
                        ]
                        # layers = collections.OrderedDict()
                        # layers["index"] = layer_index
                        # layers["values"] = [
                        #     round(x.item(), 6) for x in layer_output[i]
                        # ]
                        all_layers.append(values)

                    # out_features = collections.OrderedDict()
                    # out_features["token"] = token
                    # out_features["layers"] = all_layers
                    out_features = np.array(all_layers).mean(axis=0).tolist()
                    all_out_features.append(out_features)

                # output_json["features"] = all_out_features
                # writer.write(json.dumps(output_json) + "\n")
                vec = ' '.join(
                    ["{:.6f}".format(e) for e in pool_func(all_out_features)])
                writer.write(''.join(examples[example_index].text_a.split()) +
                             ' ' + vec + '\n')


def wordlist_to_xlmr_features(wordlist,
                              output_file,
                              model,
                              layer=-1,
                              pool='sum'):
    # Load multilingual XLM-RoBERTa Model
    xlmr = torch.hub.load('pytorch/fairseq', model)
    xlmr.eval()

    with open(output_file, 'w', encoding='utf-8') as writer:
        # Write first line information
        writer.write("{:d} {:d}\n".format(len(wordlist),
                                          xlmr.args.encoder_embed_dim))

        for word in tqdm(wordlist):
            # Extract the layer's features
            tokens = xlmr.encode(word)
            all_layers = xlmr.extract_features(tokens, return_all_hiddens=True)
            layer_feature = all_layers[int(layer)].squeeze(dim=0)

            if pool == 'max':
                feature = layer_feature[1:-1].max(dim=0)
            elif pool == 'mean':
                feature = layer_feature[1:-1].mean(dim=0)
            elif pool == 'first':
                feature = layer_feature[0]
            elif pool == 'last':
                feature = layer_feature[-1]
            else:
                feature = layer_feature[1:-1].sum(dim=0)

            vec = ' '.join([
                "{:.6f}".format(e)
                for e in feature.detach().cpu().numpy().tolist()
            ])
            writer.write(''.join(word.split()) + ' ' + vec + '\n')


def wordlist_to_fasttext(wordlist, output_file, lang='en', lowercase=False):
    vectors = FastText(language=lang, aligned=True)
    with open(output_file, 'w', encoding='utf-8') as writer:
        writer.write("{} 300\n".format(len(wordlist)))
        for word in wordlist:
            tokens = [t for t in word.split()]
            if lowercase:
                tokens = [t.lower() for t in tokens]

            vecs = [vectors[t] for t in tokens]
            n_vec = sum(1 for v in vecs if v.sum() != 0)
            feature = sum(vecs) / (n_vec if n_vec != 0 else 1)
            vec = ' '.join([
                "{:.6f}".format(e)
                for e in feature.detach().cpu().numpy().tolist()
            ])
            writer.write(''.join(word.split()) + ' ' + vec + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    # layer_indexes = [int(x) for x in args.layers.split(",")]

    if "bert" in args.model:
        if args.local_rank == -1 or not args.cuda:
            device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.
                                  is_available() and args.cuda else "cpu")
            n_gpu = 0
        else:
            device = torch.device("cuda", args.local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
            # logger.info("device: {} n_gpu: {} distributed training: {}".format(
            # device, n_gpu, bool(args.local_rank != -1)))

        examples = []
        for f in args.input_file:
            examples += read_examples(f)
        write_features_from_examples(examples,
                                     output_file=args.output_file,
                                     bert_model=args.model,
                                     layer=args.layer,
                                     device=device,
                                     batch_size=args.batch_size,
                                     max_tokens=args.max_tokens,
                                     tokenized=False,
                                     local_rank=args.local_rank,
                                     n_gpu=n_gpu,
                                     lowercase=args.lowercase,
                                     pool=args.pool)
    else:
        # Read lines into word list
        wordlist = []
        for infile in args.input_file:
            with open(infile, 'r', encoding='utf-8') as f:
                wordlist += [w.rstrip() for w in f.readlines()]

        if "xlmr" in args.model:
            wordlist_to_xlmr_features(wordlist,
                                      args.output_file,
                                      args.model,
                                      layer=args.layer,
                                      pool=args.pool)
        else:  # fastText
            wordlist_to_fasttext(wordlist,
                                 args.output_file,
                                 args.lang,
                                 lowercase=args.lowercase)


if __name__ == "__main__":
    main()
