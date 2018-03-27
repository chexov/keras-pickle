#!/usr/bin/env python
# encoding: utf-8

import os
import time
from collections import deque
from functools import lru_cache

import gzip
import pickle
from keras.utils import Sequence, OrderedEnqueuer


class PickeledSequence(Sequence):

    def __init__(self, pickles_dir: str, batches_per_pickle: int = 10):
        self.pickle_size = batches_per_pickle
        self.pickles_dir = pickles_dir
        self.pickles = []

        for r, d, f in os.walk(pickles_dir):
            for file in f:
                if file.endswith(".pickle"):
                    self.pickles.append(os.path.join(pickles_dir, file))
                elif file.endswith(".pickle.gz"):
                    self.pickles.append(os.path.join(pickles_dir, file))

    def __getitem__(self, index):
        n = index % self.pickle_size
        pickle_number = (int(index / self.pickle_size) + 1) * self.pickle_size - 1
        bb = self.pickle_from_cache(pickle_number)
        return bb[n]

    @lru_cache(maxsize=2)
    def pickle_from_cache(self, pickle_number: int):
        bn = self.pickle_fn(pickle_number)
        fn = os.path.join(self.pickles_dir, bn)
        return self.load(fn)

    def load(self, fn: str):
        if fn.endswith(".gz"):
            _f = gzip.open(fn)
        else:
            _f = open(fn, "rb")
        bb = pickle.load(_f)
        return bb

    @staticmethod
    def pickle_fn(pickle_number):
        return "b%06d.pickle" % pickle_number

    def __len__(self):
        return len(self.pickles) * self.pickle_size

    def save(self, bb: deque, pickle_number: int, compress: bool = False):
        pfn = self.pickles_dir + PickeledSequence.pickle_fn(pickle_number)

        if compress:
            pfn = pfn + ".gz"
            with gzip.open(pfn, "wb") as f:
                pickle.dump(bb, f)
        else:
            with open(pfn, "wb") as f:
                pickle.dump(bb, f)
        return pfn

    def save_pickles(self, seq: Sequence, epochs: int, pickle_size: int,
                     use_multiprocessing: bool = True,
                     shuffle: bool = True,
                     workers: int = 3,
                     max_queue_size: int = 10, compress=False):
        enqueuer = OrderedEnqueuer(seq, use_multiprocessing=use_multiprocessing, shuffle=shuffle)
        enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        generator = enqueuer.get()
        start = time.time()

        bb = deque()
        for bn in range(0, len(seq) * epochs):
            xy = next(generator)
            bb.append(xy)

            if len(bb) == pickle_size:
                outfile = self.save(bb, bn, compress)
                bb.clear()
                print("[%s] Pickled %d batches in %f sec to %s" % (bn, pickle_size, time.time() - start, outfile))
                start = time.time()
