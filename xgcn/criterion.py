# -*- coding: utf-8 -*-
#
# criterion.py - parallal GCN loss function interfacing with fairseq
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

import math
import numpy as np
from scipy.spatial.distance import cdist

import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('dictionary')
class DictionaryCriterion(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.gamma = args.gamma
        self.k = args.k[0]

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, poz_loss, neg_loss, ncorrect = self.compute_loss(
            model, net_output, sample, sample['valid_split'], reduce=reduce)
        sample_size = sample['nsentences']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(loss.data) if reduce else loss.data,
            'poz_loss': utils.item(poz_loss.data) if reduce else poz_loss.data,
            'neg_loss': utils.item(neg_loss.data) if reduce else neg_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'nvalid': len(sample['valid_split']['valid_indices']),
            'sample_size': sample['ntokens'],
            'ncorrect': ncorrect,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, valid_split,
                     reduce=True):
        src_out, tgt_out = net_output
        train_idx = valid_split['train_indices']
        valid_idx = valid_split['valid_indices']

        # Get loss
        poz_loss = torch.abs(src_out[train_idx] -
                             tgt_out[train_idx]).sum(dim=1).mean()
        shuffle = torch.randperm(len(train_idx))
        neg_loss = torch.abs(src_out[train_idx] -
                             tgt_out[train_idx][shuffle]).sum(dim=1).mean()
        loss = poz_loss - neg_loss + self.gamma

        # Calculate number of correct
        dist = torch.cdist(src_out.detach()[valid_idx],
                           tgt_out.detach()[valid_idx],
                           p=1)
        indices = dist.topk(k=self.k, dim=1, largest=False,
                            sorted=True).indices
        correct = torch.arange(len(valid_idx)).to(indices.device)
        ncorrect = torch.sum(torch.any(indices == correct.unsqueeze(1),
                                       dim=1)).cpu().numpy()
        return loss, poz_loss, neg_loss, ncorrect

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        poz_loss_sum = sum(log.get('poz_loss', 0) for log in logging_outputs)
        neg_loss_sum = sum(log.get('neg_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        nvalid = sum(log.get('nvalid', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'poz_loss': poz_loss_sum / sample_size / math.log(2),
            'neg_loss': neg_loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            agg_output.update(accuracy=ncorrect / nvalid)

        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
