# -*- coding: utf-8 -*-
#
# data_utils.py - utility functions and dataset iterators
#
# Copyright (c) 2017-present, Facebook, Inc.
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

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.data import data_utils, FairseqDataset, iterators
from fairseq.checkpoint_utils import save_checkpoint

from sklearn.model_selection import train_test_split


def Embedding(num_embeddings, embedding_dim, padding_idx, eos_index):
    """Get Embedding object with padding and eos weights set to zero. """
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    nn.init.constant_(m.weight[eos_index], 0)
    return m


def build_embedding(dictionary, embed_dim, path=None):
    padding_idx = dictionary.pad()
    eos_index = dictionary.eos()
    emb = Embedding(len(dictionary), embed_dim, padding_idx, eos_index)
    # if provided, load from preloaded dictionaries
    if path:
        embed_dict = parse_embedding(path)
        utils.load_embedding(embed_dict, dictionary, emb)
    return emb


def parse_embedding(embed_path):
    """Parse embedding text file into a dictionary of word and embedding tensors.

    The first line can have vocabulary size and dimension. The following lines
    should contain word and embedding separated by spaces.

    Example:
        2 5
        the -0.0230 -0.0264  0.0287  0.0171  0.1403
        at -0.0395 -0.1286  0.0275  0.0254 -0.0932
    """
    embed_dict = {}
    with open(embed_path, encoding='utf-8') as f_embed:
        next(f_embed)  # skip header
        for line in f_embed:
            pieces = line.rstrip().split(" ")
            embed_dict[pieces[0]] = torch.Tensor(
                [float(weight) for weight in pieces[1:]])
    return embed_dict


def gen_adjacency(samples, sparse=False):
    edges = [[i, int(neigh)] for i, s in enumerate(samples)
             for neigh in s['alignment']]
    i = torch.tensor(edges).t()
    v = torch.ones(i.size(1))
    adj = torch.sparse.FloatTensor(i, v)
    if not sparse:
        adj = adj.to_dense()
    return adj


def collate(samples,
            pad_idx=1,
            eos_idx=2,
            left_pad_source=False,
            left_pad_target=False,
            input_feeding=False,
            test_size=0.1,
            seed=215):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens([s[key] for s in samples], pad_idx,
                                         eos_idx, left_pad,
                                         move_eos_to_beginning)

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() - 1 for s in samples])
    # src_lengths, sort_order = src_lengths.sort(descending=True)
    # id = id.index_select(0, sort_order)
    # src_tokens = src_tokens.index_select(0, sort_order)

    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        # target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s['target'].numel() - 1 for s in samples])
        ntokens = sum(len(s['target']) for s in samples)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    # Generate adjacency matrices
    adj = None
    if samples[0].get('alignment', None) is not None:
        # adj = torch.FloatTensor(adjacency(samples))
        adj = gen_adjacency(samples)

    train_idx, valid_idx = train_test_split(range(len(samples)),
                                            test_size=test_size,
                                            random_state=seed)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'tgt_tokens': target,
            'adj': adj,
            'src_lengths': src_lengths,
            'tgt_lengths': tgt_lengths,
        },
        'valid_split': {
            'train_indices': train_idx,
            'valid_indices': valid_idx,
        },
    }

    return batch


def get_epoch_iterator(task,
                       dataset,
                       max_tokens=None,
                       max_sentences=None,
                       max_positions=None,
                       ignore_invalid_inputs=False,
                       required_batch_size_multiple=1,
                       num_workers=0,
                       seed=215,
                       num_shards=1,
                       shard_id=0,
                       epoch=0):
    """ Get an iterator that yields batches of data from the given dataset. """
    if dataset in task.dataset_to_epoch_iter:
        return task.dataset_to_epoch_iter[dataset]

    # initialize the dataset with the correct starting epoch
    dataset.set_epoch(epoch)

    # get indices ordered by example size
    with data_utils.numpy_seed(seed):
        indices = dataset.ordered_indices()

    assert isinstance(dataset, FairseqDataset)

    # get indices ordered by example size
    with data_utils.numpy_seed(seed):
        indices = dataset.ordered_indices()

    # filter examples that are too large
    if max_positions is not None:
        indices = data_utils.filter_by_size(
            indices,
            dataset,
            max_positions,
            raise_exception=(not ignore_invalid_inputs))

    # create mini-batches with given size constraints
    batch_sampler = data_utils.batch_by_size(
        indices,
        dataset.num_tokens,
        max_tokens=max_tokens,
        max_sentences=max_sentences,
        required_batch_size_multiple=required_batch_size_multiple)

    epoch_iter = iterators.EpochBatchIterator(dataset=dataset,
                                              collate_fn=collate,
                                              batch_sampler=batch_sampler,
                                              seed=seed,
                                              num_shards=num_shards,
                                              shard_id=shard_id,
                                              num_workers=num_workers,
                                              epoch=epoch)
    task.dataset_to_epoch_iter[dataset] = epoch_iter
    return epoch_iter


def load_checkpoint(args, task, model, trainer, **passthrough_args):
    """
    Load a checkpoint and restore the training iterator.
    *passthrough_args* will be passed through to
    ``trainer.get_train_iterator``.
    """
    # only one worker should attempt to create the required dir
    if args.distributed_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.restore_file == 'checkpoint_last.pt':
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint_last.pt')
    else:
        checkpoint_path = args.restore_file

    extra_state = trainer.load_checkpoint(checkpoint_path,
                                          args.reset_optimizer,
                                          args.reset_lr_scheduler,
                                          eval(args.optimizer_overrides),
                                          reset_meters=args.reset_meters)

    if (extra_state is not None and 'best' in extra_state
            and not args.reset_optimizer and not args.reset_meters):
        save_checkpoint.best = extra_state['best']

    if extra_state is not None and not args.reset_dataloader:
        # restore iterator from checkpoint
        itr_state = extra_state['train_iterator']
        epoch_itr = get_epoch_iterator(
            task,
            task.dataset(args.train_subset),
            max_tokens=args.max_tokens,
            max_sentences=None,
            max_positions=utils.resolve_max_positions(task.max_positions(),
                                                      model.max_positions()),
            ignore_invalid_inputs=True,
            num_workers=args.num_workers,
            seed=args.seed,
            epoch=itr_state['epoch'])
        epoch_itr.load_state_dict(itr_state)
    else:
        epoch_itr = get_epoch_iterator(
            task,
            task.dataset(args.train_subset),
            max_tokens=args.max_tokens,
            max_sentences=None,
            max_positions=utils.resolve_max_positions(task.max_positions(),
                                                      model.max_positions()),
            ignore_invalid_inputs=True,
            num_workers=args.num_workers,
            seed=args.seed,
            epoch=0)

    trainer.lr_step(epoch_itr.epoch)
    return extra_state, epoch_itr
