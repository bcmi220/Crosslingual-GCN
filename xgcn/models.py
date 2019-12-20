# -*- coding: utf-8 -*-
#
# models.py - parallax GCN models and layers interfacing with fairseq
#
# Copyright (c) 2017 Thomas Kipf.
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
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (FairseqEncoder, FairseqDecoder, BaseFairseqModel,
                            FairseqMultiModel, register_model,
                            register_model_architecture)
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask

from .data_utils import Embedding, build_embedding


@register_model('xgcn')
class XGCNModel(BaseFairseqModel):
    """ Parallax Graph Convolutional Network model for bilingual word
    translation.

    Args:
        encoders (LinearEncoder, LinearEncoder): tuple of source and target
            linear encoders.
        decoders (GCNDecoder, GCNDecoder): tuple of source and target
            linear encoders.
    """
    def __init__(self, src_encoder, tgt_encoder):
        super().__init__()
        self.src_encoder = src_encoder
        self.tgt_encoder = tgt_encoder
        assert isinstance(self.src_encoder, FairseqEncoder)
        assert isinstance(self.tgt_encoder, FairseqEncoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--gamma',
                            type=float,
                            metavar='GAMMA',
                            help="""margin between positive and negative
                                    samples""")
        parser.add_argument('--k',
                            type=int,
                            metavar='K',
                            nargs='+',
                            help='Precision @ k matches')
        parser.add_argument('--embed-dim',
                            type=int,
                            metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--layers',
                            type=int,
                            metavar='N',
                            help='num layers')
        parser.add_argument('--hidden-dim',
                            type=int,
                            metavar='N',
                            help='hidden dimension for GCN')
        parser.add_argument('--output-dim',
                            type=int,
                            metavar='N',
                            help='output dimension for GCN')
        parser.add_argument('--bias',
                            action='store_true',
                            help='bias in graph convolutional layer')
        parser.add_argument('--share-encoders',
                            action='store_true',
                            help="""share graph convolution layers
                                    across languages""")
        parser.add_argument('--source-embed-path',
                            type=str,
                            metavar='STR',
                            help="""path to pre-trained source language
                                    embedding""")
        parser.add_argument('--target-embed-path',
                            type=str,
                            metavar='STR',
                            help="""path to pre-trained target language
                                    embedding""")
        parser.add_argument('--share-all-embed',
                            action='store_true',
                            help='share encoder, decoder and output embeddings'
                            ' (requires shared dictionary and embed dim)')

    def forward(self,
                src_tokens,
                tgt_tokens,
                adj=None,
                src_lengths=None,
                tgt_lengths=None):
        """ Run the forward pass for an XGCN model. """
        src_out = self.src_encoder(src_tokens, src_lengths, adj)
        tgt_out = self.tgt_encoder(tgt_tokens, tgt_lengths, adj)
        return src_out, tgt_out

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_encoders:
            args.share_all_embed = True

        # build (shared) semantic embeddings
        if args.share_all_embed:
            if src_dict != tgt_dict:
                raise ValueError(
                    '--share-all-embeddings requires a joined dictionary')
            if args.embed_dim != args.embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim'
                )
            src_embed_tokens = FairseqMultiModel.build_shared_embed(
                dicts=src_dict,
                langs=task.langs,
                embed_dim=args.embed_dim,
                build_embedding=build_embedding,
                pretrained_embed_path=args.source_embed_path)
            tgt_embed_tokens = src_embed_tokens
        else:
            src_embed_tokens = build_embedding(src_dict, args.embed_dim,
                                               args.source_embed_path)
            tgt_embed_tokens = build_embedding(tgt_dict, args.embed_dim,
                                               args.target_embed_path)

        # Shared encoders/decoders (if applicable)
        src_encoder = GCNEncoder(args, src_dict, src_embed_tokens)
        tgt_encoder = src_encoder if args.share_encoders else GCNEncoder(
            args, tgt_dict, tgt_embed_tokens)

        return cls(src_encoder, tgt_encoder)


class GCNEncoder(FairseqEncoder):
    """ Layers of GCN layers. """
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.embed_tokens = embed_tokens

        self.layers = nn.ModuleList([])
        dimensions = [
            args.embed_dim
        ] + (args.layers - 1) * [args.hidden_dim] + [args.output_dim]
        self.layers.extend([
            GraphConvolutionalLayer(dimensions[i],
                                    dimensions[i + 1],
                                    bias=args.bias) for i in range(args.layers)
        ])

        self.register_buffer('version', torch.Tensor([2]))
        # self.normalize = args.encoder_normalize_steps

    def forward(self, tokens, lengths, adj=None):
        embeds = self.embed_tokens(tokens)
        mask = torch.arange(embeds.size(1)).expand(
            embeds.size(0), embeds.size(1)).to(
                lengths.device) < lengths.unsqueeze(1)
        out = torch.sum(embeds * mask.unsqueeze(2),
                        dim=1) / lengths.unsqueeze(1)

        if adj is not None:  # no GCN for validation
            for layer in self.layers:
                out = layer(out, adj)

        return out


class GraphConvolutionalLayer(nn.Module):
    """ Simple GCN layer. """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        torch.nn.init.xavier_uniform_(self.linear.weight)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            stdv = 1. / math.sqrt(self.linear.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj):
        output = adj.mm(self.linear(input))
        if self.bias is not None:
            output += self.bias
        return output


@register_model_architecture('xgcn', 'xgcn')
def base_architecture(args):
    args.gamma = getattr(args, 'gamma', 0)
    args.k = getattr(args, 'k', [10])
    args.embed_dim = getattr(args, 'embed_dim', 300)
    args.layers = getattr(args, 'layers', 2)
    args.hidden_dim = getattr(args, 'hidden_dim', 200)
    args.output_dim = getattr(args, 'output_dim', 100)
    args.bias = getattr(args, 'bias', True)
    args.share_encoders = getattr(args, 'share_encoders', False)
    args.source_embed_path = getattr(args, 'source_embed_path', None)
    args.target_embed_path = getattr(args, 'target_embed_path', None)
    args.share_all_embed = getattr(args, 'share_all_embed', False)
