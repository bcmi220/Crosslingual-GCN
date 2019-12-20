# -*- coding: utf-8 -*-
#
# generate_mesh.py - train/test split from extracted MeSH terms
#
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
import numpy as np
import scipy.stats as stats
from os.path import join
from collections import defaultdict
from sklearn.model_selection import train_test_split


def add_args(parser):
    # fmt: off
    parser.add_argument("-s",
                        "--source-langs",
                        required=True,
                        metavar="LANG",
                        nargs='+',
                        help="list of source languages")
    parser.add_argument("-t",
                        "--target-langs",
                        required=True,
                        metavar="LANG",
                        nargs='+',
                        help="list of target languages")
    parser.add_argument("--file-pre",
                        metavar="FP",
                        required=True,
                        help="file prefix")
    parser.add_argument("--dest-dir",
                        metavar="DIR",
                        default="data-bin",
                        help="destination dir")
    parser.add_argument("--test-size",
                        metavar="DIR",
                        type=float,
                        default="0.1",
                        help="size of test/valid set to train")
    return parser


def main(args):
    for sl in args.source_langs:
        for tl in args.target_langs:
            sf = open(args.file_pre + '.' + sl, 'r')
            tf = open(args.file_pre + '.' + tl, 'r')
            tn = open(args.file_pre + '.tree.num', 'r')

            source = []
            target = []
            tree = []
            # tree_dict = {}
            # i = 0
            for s, t, n in zip(sf, tf, tn):
                if s != '\n' and t != '\n':  # only take cases where both exist
                    if tl == "RUS":
                        t = t.capitalize()
                    source.append(s.rstrip())
                    target.append(t.rstrip())
                    tree.append(n.rstrip())
                    # for num in n.split(','):
                    #     tree_dict[num.rstrip()] = i
                    # i += 1

            sf.close()
            tf.close()
            tn.close()

            # Dataset split
            train_s, test_s, train_t, test_t, train_n, test_n = train_test_split(
                source,
                target,
                tree,
                test_size=args.test_size,
                random_state=215)

            # train_s, valid_s, train_t, valid_t, train_n, valid_n = train_test_split(
            #     train_s,
            #     train_t,
            #     train_n,
            #     test_size=args.test_size,
            #     random_state=215)

            def treenum2edge(treenum):
                tree_dict = {
                    n: i
                    for i, tn in enumerate(treenum) for n in tn.split(',')
                }

                # Process tree number to generate edge lists
                edge = [set() for _ in range(len(treenum))]
                for i, tns in enumerate(treenum):
                    for tn in tns.split(','):
                        tn_sep = tn.split('.')
                        while len(tn_sep) > 1:
                            tn_sep = tn_sep[:-1]
                            ancestor = '.'.join(tn_sep)
                            if ancestor in tree_dict:
                                ans = tree_dict[ancestor]
                                edge[ans].add(i)
                                edge[i].add(ans)
                                break
                return edge

            train_e = treenum2edge(train_n)
            # valid_e = treenum2edge(valid_n)
            test_e = treenum2edge(test_n)

            # Write output files
            train_path = join(args.dest_dir, 'train.' + sl + '-' + tl)
            # valid_path = join(args.dest_dir, 'valid.' + sl + '-' + tl)
            test_path = join(args.dest_dir, 'test.' + sl + '-' + tl)

            with open(train_path + '.' + sl, 'w') as fp:
                fp.write('\n'.join(train_s))
            with open(train_path + '.' + tl, 'w') as fp:
                fp.write('\n'.join(train_t))
            with open(train_path + '.dict', 'w') as fp:
                for s, t in zip(train_s, train_t):
                    fp.write("{} {}\n".format(''.join(s.split()),
                                              ''.join(t.split())))
            with open(
                    join(args.dest_dir, 'train.' + tl + '-' + sl) + '.dict',
                    'w') as fp:
                for s, t in zip(train_s, train_t):
                    fp.write("{} {}\n".format(''.join(t.split()),
                                              ''.join(s.split())))
            with open(train_path + '.edge', 'w') as fp:
                fp.write('\n'.join(' '.join(str(e) for e in edge)
                                   for edge in train_e))
            with open(train_path + '.len', 'w') as fp:
                fp.write('\n'.join("{:d}".format(
                    stats.mode([len(n.split('.'))
                                for n in tn.split(',')]).mode[0])
                                   for tn in train_n))

            # with open(valid_path + '.' + sl, 'w') as fp:
            #     fp.write('\n'.join(valid_s))
            # with open(valid_path + '.' + tl, 'w') as fp:
            #     fp.write('\n'.join(valid_t))
            # with open(valid_path + '.edge', 'w') as fp:
            #     fp.write('\n'.join(' '.join(str(e) for e in edge)
            #                        for edge in valid_e))
            # with open(
            #         join(args.dest_dir, 'valid.' + tl + '-' + sl) + '.dict',
            #         'w') as fp:
            #     for s, t in zip(valid_s, valid_t):
            #         fp.write("{} {}\n".format(''.join(t.split()),
            #                                   ''.join(s.split())))
            # with open(valid_path + '.edge', 'w') as fp:
            #     fp.write('\n'.join(' '.join(str(e) for e in edge)
            #                        for edge in valid_e))
            # with open(valid_path + '.len', 'w') as fp:
            #     fp.write('\n'.join("{:d}".format(
            #         stats.mode([len(n.split('.'))
            #                     for n in tn.split(',')]).mode[0])
            #                        for tn in valid_n))

            with open(test_path + '.' + sl, 'w') as fp:
                fp.write('\n'.join(test_s))
            with open(test_path + '.' + tl, 'w') as fp:
                fp.write('\n'.join(test_t))
            with open(test_path + '.dict', 'w') as fp:
                for s, t in zip(test_s, test_t):
                    fp.write("{} {}\n".format(''.join(s.split()),
                                              ''.join(t.split())))
            with open(
                    join(args.dest_dir, 'test.' + tl + '-' + sl) + '.dict',
                    'w') as fp:
                for s, t in zip(test_s, test_t):
                    fp.write("{} {}\n".format(''.join(t.split()),
                                              ''.join(s.split())))
            with open(test_path + '.edge', 'w') as fp:
                fp.write('\n'.join(' '.join(str(e) for e in edge)
                                   for edge in test_e))
            with open(test_path + '.len', 'w') as fp:
                fp.write('\n'.join("{:d}".format(
                    stats.mode([len(n.split('.'))
                                for n in tn.split(',')]).mode[0])
                                   for tn in test_n))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)
