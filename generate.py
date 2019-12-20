# -*- coding: utf-8 -*-
#
# generate.py - translate pre-processed data with a trained model
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

import torch

from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.data import Dictionary
from fairseq.meters import StopwatchMeter, TimeMeter

from xgcn import XGCNModel, data_utils


def add_gen_args(parser):
    parser.add_argument("-o",
                        "--output-file",
                        metavar="FILE",
                        help="output file path")
    parser.add_argument("--reverse",
                        action='store_true',
                        help="""Reverse source and target langs in
                                evaluation.""")
    return parser


def main(args):
    assert args.path is not None, '--path required for generation!'
    utils.import_user_module(args)
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset, combine=False, epoch=0)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task)

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load dataset (possibly sharded)
    itr = data_utils.get_epoch_iterator(
        task,
        task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=None,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        num_workers=args.num_workers,
        seed=args.seed).next_epoch_itr(shuffle=False)

    # Initialize gen_timer
    gen_timer = StopwatchMeter()
    with progress_bar.build_progress_bar(args, itr) as t:
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue
            src_out, tgt_out = [], []

            gen_timer.start()
            for model in models:
                model.eval()
                with torch.no_grad():
                    s_out, t_out = model.forward(**sample['net_input'])
                    src_out.append(s_out)
                    tgt_out.append(t_out)
            gen_timer.stop()

            src_out = sum(src_out) / len(src_out)
            tgt_out = sum(tgt_out) / len(tgt_out)

            valid_idx = sample['valid_split']['valid_indices']
            dist = torch.cdist(src_out.detach()[valid_idx],
                               tgt_out.detach()[valid_idx],
                               p=1)

            fo, fl = None, None
            if args.output_file:
                fo = open(args.output_file, 'w', encoding='utf-8')
                fo.write("k acc\n")
            if args.log_file:
                fl = open(args.log_file, 'w', encoding='utf-8')

            # Load MUSE embeddings if available
            src_embed, tgt_embed = None, None
            if args.source_embed_path and args.target_embed_path:
                with open(args.source_embed_path, 'r') as f:
                    first_line = f.readline()
                embed_dim = int(first_line.rstrip().split()[1])

                src_d = task.source_dictionary
                src_symbol = [
                    "".join(
                        src_d.string(
                            utils.strip_pad(
                                sample['net_input']['src_tokens'][i, :],
                                src_d.pad()), args.remove_bpe).split())
                    for i in range(sample['nsentences'])
                ]
                tgt_d = task.target_dictionary
                tgt_symbol = [
                    "".join(
                        tgt_d.string(
                            utils.strip_pad(
                                sample['net_input']['tgt_tokens'][i, :],
                                tgt_d.pad()), args.remove_bpe).split())
                    for i in range(sample['nsentences'])
                ]

                def build_dict(symbol_list):
                    d = Dictionary()
                    for symbol in symbol_list:
                        d.add_symbol(symbol)
                    return d

                src_dict = build_dict(src_symbol)
                tgt_dict = build_dict(tgt_symbol)

                src_embed = data_utils.build_embedding(
                    src_dict, embed_dim, path=args.source_embed_path)
                tgt_embed = data_utils.build_embedding(
                    tgt_dict, embed_dim, path=args.target_embed_path)

                src_sem_out = [
                    src_embed(torch.tensor(src_dict.index(s))).unsqueeze(0)
                    for s in src_symbol
                ]
                src_sem_out = torch.cat(src_sem_out, dim=0).to(dist.device)
                tgt_sem_out = [
                    tgt_embed(torch.tensor(tgt_dict.index(s))).unsqueeze(0)
                    for s in tgt_symbol
                ]
                tgt_sem_out = torch.cat(tgt_sem_out, dim=0).to(dist.device)

                sdist = torch.cdist(src_sem_out[valid_idx],
                                    tgt_sem_out[valid_idx],
                                    p=1)
                dim = src_out.size(1)

                for k in args.k:
                    best_acc = -1
                    best_b = -1
                    for b in [1e-4 * t for t in range(10001)]:
                        fdist = dist / dim * b + sdist / embed_dim * (1 - b)
                        if args.reverse:
                            fdist = fdist.t()
                        indices = fdist.topk(k=k,
                                             dim=1,
                                             largest=False,
                                             sorted=True).indices
                        correct = torch.arange(len(valid_idx)).to(
                            indices.device)
                        ncorrect = torch.sum(
                            torch.any(indices == correct.unsqueeze(1),
                                      dim=1)).cpu().numpy()
                        accuracy = ncorrect / len(valid_idx)
                        if accuracy > best_acc:
                            best_acc = accuracy
                            best_b = b

                    print('| Accuracy at k={:d}, beta={:.2f} : {:.5f}'.format(
                        k, best_b, best_acc))
            else:
                for k in args.k:
                    if args.reverse:
                        dist = dist.t()
                    indices = dist.topk(k=k, dim=1, largest=False,
                                        sorted=True).indices
                    correct = torch.arange(len(valid_idx)).to(indices.device)
                    ncorrect = torch.sum(
                        torch.any(indices == correct.unsqueeze(1),
                                  dim=1)).detach().cpu().numpy()
                    accuracy = ncorrect / len(valid_idx)

                    if fo:
                        fo.write("{:d} {:.6f}\n".format(k, accuracy))
                    print('| Accuracy at k={}: {:.5f}'.format(k, accuracy))


def cli_main():
    parser = options.get_generation_parser()
    XGCNModel.add_args(parser)
    parser = add_gen_args(parser)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
