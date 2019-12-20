# -*- coding: utf-8 -*-
#
# preprocess.py - build vocabularies and binarize training data
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
import shutil
from collections import Counter
import multiprocessing
multiprocessing.set_start_method('spawn', True)

import torch

from fairseq import options, tasks, utils
from fairseq.data import indexed_dataset
from fairseq.binarizer import Binarizer
from fairseq import options

from tools.extract_features import (InputExample, write_features_from_examples,
                                    wordlist_to_xlmr_features, tokenize)


def add_preprocess_args(parser):
    group = parser.add_argument_group('Preprocessing')
    group.add_argument("-s",
                       "--source-langs",
                       default=None,
                       metavar="LANG",
                       nargs='+',
                       help="list of source languages")
    group.add_argument("-t",
                       "--target-langs",
                       default=None,
                       metavar="LANG",
                       nargs='+',
                       help="list of target languages")
    group.add_argument("--train-pre",
                       metavar="FP",
                       default=None,
                       help="train file prefix")
    group.add_argument("--valid-pre",
                       metavar="FP",
                       default=None,
                       help="valid file prefixes")
    group.add_argument("--test-pre",
                       metavar="FP",
                       default=None,
                       help="test file prefixes")
    group.add_argument("--align-suffix",
                       metavar="FP",
                       default="",
                       help="alignment file suffix")
    group.add_argument("--out-pre",
                       metavar="OP",
                       default="",
                       help="output file prefix")
    group.add_argument("--dest-dir",
                       metavar="DIR",
                       default="data-bin",
                       help="destination dir")
    group.add_argument("--threshold",
                       metavar="N",
                       default=0,
                       type=int,
                       help="""map words appearing less than
                               threshold times to unknown""")
    group.add_argument("--nwords",
                       metavar="N",
                       default=-1,
                       type=int,
                       help="number of words to retain")
    parser.add_argument('--dataset-impl',
                        metavar='FORMAT',
                        default='cached',
                        choices=['raw', 'lazy', 'cached', 'mmap'],
                        help='output dataset implementation')
    group.add_argument("--join-dict",
                       action="store_true",
                       help="Generate joined dictionary")
    group.add_argument("--only-source",
                       action="store_true",
                       help="Only process the source language")
    group.add_argument("--padding-factor",
                       metavar="N",
                       default=8,
                       type=int,
                       help="Pad dictionary size to be multiple of N")
    group.add_argument("--workers",
                       metavar="N",
                       default=1,
                       choices=[1],
                       type=int,
                       help="number of parallel workers")
    group.add_argument("--model",
                       default="bert-base-multilingual-cased",
                       type=str,
                       help="""Bert pre-trained model or
                               XLM-R (XLM-RoBERTa) pre-trained model""")
    group.add_argument("--lowercase",
                       action='store_true',
                       help="""Set this flag if you are using an
                                uncased model.""")
    group.add_argument("--layer",
                       default="-2",
                       type=int,
                       help="""Layer to be extracted from BERT.""")
    group.add_argument("--batch-size",
                       default=32,
                       type=int,
                       help="Batch size for predictions.")
    group.add_argument("--local-rank",
                       type=int,
                       default=-1,
                       help="local_rank for distributed training on gpus")
    group.add_argument("--cuda",
                       action='store_true',
                       help="Whether not to use CUDA when available")
    group.add_argument('--pool',
                       type=str,
                       metavar='POOL',
                       help="""Pooling function for term embeddings.""")

    return parser


def binarize(args,
             filename,
             vocab,
             output_prefix,
             src_lang,
             tgt_lang,
             lang,
             offset,
             end,
             append_eos=True):
    ds = indexed_dataset.make_builder(dataset_dest_file(
        args, output_prefix, src_lang, tgt_lang, lang, "bin"),
                                      impl=args.dataset_impl,
                                      vocab_size=len(vocab))

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(filename,
                             vocab,
                             consumer,
                             append_eos=append_eos,
                             offset=offset,
                             end=end)
    ds.finalize(
        dataset_dest_file(args, output_prefix, src_lang, tgt_lang, lang,
                          "idx"))
    return res


def binarize_alignments(args, filename, parse_alignment, output_prefix, src,
                        tgt, offset, end):
    ds = indexed_dataset.make_builder(dataset_dest_file(
        args, output_prefix, src, tgt, None, "bin"),
                                      impl=args.dataset_impl,
                                      vocab_size=None)

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize_alignments(filename,
                                        parse_alignment,
                                        consumer,
                                        offset=offset,
                                        end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, src, tgt, None, "idx"))
    return res


def dataset_dest_prefix(args, output_prefix, src_lang, tgt_lang, lang=None):
    base = "{}/{}".format(args.dest_dir, output_prefix)
    if lang is not None:
        lang_part = ".{}-{}.{}".format(src_lang, tgt_lang, lang)
    elif args.only_source:
        lang_part = ""
    else:
        lang_part = ".{}-{}".format(src_lang, tgt_lang)

    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, src_lang, tgt_lang, lang,
                      extension):
    base = dataset_dest_prefix(args, output_prefix, src_lang, tgt_lang, lang)
    return "{}.{}".format(base, extension)


def get_offsets(input_file, num_workers):
    return Binarizer.find_offsets(input_file, num_workers)


def main(args):
    utils.import_user_module(args)
    print(args)
    os.makedirs(args.dest_dir, exist_ok=True)
    target = not args.only_source
    task = tasks.get_task(args.task)
    all_langs = list(set(args.source_langs + args.target_langs))

    def train_path(src_lang, tgt_lang, lang, prefix=args.train_pre, tok=None):
        path = "{}.{}-{}{}".format(prefix, src_lang, tgt_lang,
                                   ("." + lang) if lang else "")
        if tok:
            path += ".tok"
        return path

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        if type(lang) == list:
            lang = '-'.join(sorted(list(set(lang))))
        return os.path.join(args.dest_dir,
                            file_name(args.out_pre + prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    def features_path(feature_pre, lang):
        return dest_path(feature_pre, lang) + ".txt"

    def build_dictionary(filenames):
        # assert src ^ tgt
        return task.build_dictionary(
            filenames,
            workers=args.workers,
            threshold=args.threshold,
            nwords=args.nwords,
            padding_factor=args.padding_factor,
        )

    def tokenize_file(prefix):
        if prefix:
            input_path = train_path(sl, tl, sl, prefix=prefix)
            tokenize(input_path,
                     input_path + '.tok',
                     model=args.model,
                     lowercase=args.lowercase)
            input_path = train_path(sl, tl, tl, prefix=prefix)
            tokenize(input_path,
                     input_path + '.tok',
                     model=args.model,
                     lowercase=args.lowercase)

    for sl in args.source_langs:
        # if os.path.exists(dict_path(sl)):
        #     raise FileExistsError(dict_path(sl))
        for tl in args.target_langs:
            # if os.path.exists(dict_path(tl)):
            #     raise FileExistsError(dict_path(tl))
            if sl == tl:
                raise ValueError(
                    "Source language and target language lists cannot overlap."
                )
            if args.model:
                for pref in (args.train_pre, args.valid_pre, args.test_pre):
                    tokenize_file(pref)

    if args.join_dict:
        joined_dict = build_dictionary({
            train_path(sl, tl, sl, tok=args.model)
            for sl in args.source_langs for tl in args.target_langs
        } | {
            train_path(sl, tl, tl, tok=args.model)
            for sl in args.source_langs for tl in args.target_langs
        })
        for lang in all_langs:
            joined_dict.save(dict_path(lang))
    else:
        dicts = {}
        for sl in args.source_langs:
            dicts[sl] = build_dictionary({
                train_path(sl, tl, sl, tok=args.model)
                for tl in args.target_langs
            })
        for tl in args.target_langs:
            dicts[tl] = build_dictionary({
                train_path(sl, tl, tl, tok=args.model)
                for sl in args.source_langs
            })
        for lang, dic in dicts.items():
            dic.save(dict_path(lang))

    # Convert vocabulary to features if necessary
    def convert_dict_to_examples(dic):
        """Read a list of `InputExample`s from an input file."""
        examples = []
        unique_id = 0
        for i, sym in enumerate(dic.symbols):
            if i < dic.nspecial:
                continue
            if "madeupword" in sym:
                continue
            text_a = sym
            text_b = None
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a,
                             text_b=text_b))
            unique_id += 1
        return examples

    def dict_to_wordlist(dic):
        """Read a list of `InputExample`s from an input file."""
        wordlist = [
            sym for i, sym in enumerate(dic.symbols)
            if i >= dic.nspecial and "madeupword" not in sym
        ]
        return wordlist

    if args.local_rank == -1 or not args.cuda:
        device = torch.device("cuda:{}".format(
            args.cuda) if torch.cuda.is_available() and args.cuda else "cpu")
        n_gpu = 0
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    if args.model:
        if "bert" in args.model:
            if args.join_dict:
                examples = convert_dict_to_examples(joined_dict)
                write_features_from_examples(examples,
                                             features_path(
                                                 args.model, all_langs),
                                             args.model,
                                             args.layer,
                                             device,
                                             args.batch_size,
                                             max_tokens=3,
                                             tokenized=True,
                                             local_rank=args.local_rank,
                                             n_gpu=n_gpu,
                                             lowercase=args.lowercase,
                                             pool=args.pool)
            else:
                for lang, dic in dicts.items():
                    examples = convert_dict_to_examples(dic)
                    write_features_from_examples(examples,
                                                 features_path(
                                                     args.model, lang),
                                                 args.model,
                                                 args.layer,
                                                 device,
                                                 args.batch_size,
                                                 max_tokens=3,
                                                 tokenized=True,
                                                 local_rank=args.local_rank,
                                                 n_gpu=n_gpu,
                                                 lowercase=args.lowercase,
                                                 pool=args.pool)
        elif "xlmr" in args.model:
            if args.join_dict:
                wordlist = dict_to_wordlist(joined_dict)
                wordlist_to_xlmr_features(joined_dict,
                                          features_path(args.model, all_langs),
                                          args.model, args.layers)
            else:
                for lang, dic in dicts.items():
                    wordlist = dict_to_wordlist(dic)
                    wordlist_to_xlmr_features(wordlist,
                                              features_path(args.model, lang),
                                              args.model, args.layers)

    def make_binary_dataset(vocab, input_prefix, output_prefix, src_lang,
                            tgt_lang, lang, num_workers):
        print("| [{}] Dictionary: {} types".format(lang, len(vocab) - 1))
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        input_file = "{}.{}-{}.{}".format(input_prefix, src_lang, tgt_lang,
                                          lang)
        if args.model:
            input_file += ".tok"
        offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = multiprocessing.Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize,
                    (args, input_file, vocab, prefix, src_lang, tgt_lang, lang,
                     offsets[worker_id], offsets[worker_id + 1]),
                    callback=merge_result)
            pool.close()

        ds = indexed_dataset.make_builder(dataset_dest_file(
            args, output_prefix, src_lang, tgt_lang, lang, "bin"),
                                          impl=args.dataset_impl,
                                          vocab_size=len(vocab))
        merge_result(
            Binarizer.binarize(input_file,
                               vocab,
                               lambda t: ds.add_item(t),
                               offset=0,
                               end=offsets[1]))
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, src_lang,
                                                     tgt_lang, lang)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(
            dataset_dest_file(args, output_prefix, src_lang, tgt_lang, lang,
                              "idx"))

        print("| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
            lang,
            input_file,
            n_seq_tok[0],
            n_seq_tok[1],
            100 * sum(replaced.values()) / n_seq_tok[1],
            vocab.unk_word,
        ))

    def make_binary_alignment_dataset(input_prefix, output_prefix, src, tgt,
                                      num_workers):
        nseq = [0]

        def merge_result(worker_result):
            nseq[0] += worker_result['nseq']

        parse_alignment = lambda s: torch.IntTensor(
            [int(t) for t in s.split()])
        input_file = input_prefix
        offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = multiprocessing.Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize_alignments,
                    (args, input_file, parse_alignment, prefix, src, tgt,
                     offsets[worker_id], offsets[worker_id + 1]),
                    callback=merge_result)
            pool.close()

        ds = indexed_dataset.make_builder(dataset_dest_file(
            args, output_prefix, src, tgt, None, "bin"),
                                          impl=args.dataset_impl)

        merge_result(
            Binarizer.binarize_alignments(input_file,
                                          parse_alignment,
                                          lambda t: ds.add_item(t),
                                          offset=0,
                                          end=offsets[1]))
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, src, tgt)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(
            dataset_dest_file(args, output_prefix, src, tgt, None, "idx"))

        print("| [alignments] {}: parsed {} alignments".format(
            input_file, nseq[0]))

    def make_dataset(vocab,
                     input_prefix,
                     output_prefix,
                     src_lang,
                     tgt_lang,
                     lang,
                     num_workers=1):
        if args.dataset_impl == "raw":
            # Copy original text file to destination folder
            output_text_file = dest_path(
                output_prefix + ".{}-{}".format(src_lang, tgt_lang),
                lang,
            )
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)
        else:
            make_binary_dataset(vocab, input_prefix, output_prefix, src_lang,
                                tgt_lang, lang, num_workers)

    def make_all(src_lang, tgt_lang):
        if args.train_pre:
            make_dataset(joined_dict if args.join_dict else dicts[src_lang],
                         args.train_pre,
                         "train",
                         src_lang,
                         tgt_lang,
                         src_lang,
                         num_workers=args.workers)
            make_dataset(joined_dict if args.join_dict else dicts[tgt_lang],
                         args.train_pre,
                         "train",
                         src_lang,
                         tgt_lang,
                         tgt_lang,
                         num_workers=args.workers)
        if args.valid_pre:
            make_dataset(joined_dict if args.join_dict else dicts[src_lang],
                         args.valid_pre,
                         "valid",
                         src_lang,
                         tgt_lang,
                         src_lang,
                         num_workers=args.workers)
            make_dataset(joined_dict if args.join_dict else dicts[tgt_lang],
                         args.valid_pre,
                         "valid",
                         src_lang,
                         tgt_lang,
                         tgt_lang,
                         num_workers=args.workers)
        if args.test_pre:
            make_dataset(joined_dict if args.join_dict else dicts[src_lang],
                         args.test_pre,
                         "test",
                         src_lang,
                         tgt_lang,
                         src_lang,
                         num_workers=args.workers)
            make_dataset(joined_dict if args.join_dict else dicts[tgt_lang],
                         args.test_pre,
                         "test",
                         src_lang,
                         tgt_lang,
                         tgt_lang,
                         num_workers=args.workers)

    def make_all_alignments(src, tgt):
        if args.train_pre:
            train_align_path = args.train_pre + ".{}-{}.".format(
                src, tgt) + args.align_suffix
            make_binary_alignment_dataset(train_align_path,
                                          "train.align",
                                          src,
                                          tgt,
                                          num_workers=args.workers)
        if args.valid_pre:
            valid_align_path = args.valid_pre + ".{}-{}.".format(
                src, tgt) + args.align_suffix
            make_binary_alignment_dataset(valid_align_path,
                                          "valid.align",
                                          src,
                                          tgt,
                                          num_workers=args.workers)
        if args.test_pre:
            test_align_path = args.test_pre + ".{}-{}.".format(
                src, tgt) + args.align_suffix
            make_binary_alignment_dataset(test_align_path,
                                          "test.align",
                                          src,
                                          tgt,
                                          num_workers=args.workers)

    for src in args.source_langs:
        for tgt in args.target_langs:
            make_all(src, tgt)
            if args.align_suffix:
                make_all_alignments(src, tgt)

    print("| Wrote preprocessed data to {}".format(args.dest_dir))


if __name__ == "__main__":
    parser = options.get_parser('Preprocessing', default_task='translation')
    add_preprocess_args(parser)
    args = parser.parse_args()
    main(args)
