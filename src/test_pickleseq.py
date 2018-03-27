#!/usr/bin/env python
# encoding: utf-8
from keras.utils import Sequence

from pickleseq import PickeledSequence


class MySeq(Sequence):
    def __getitem__(self, index):
        return 1

    def __len__(self):
        return 1


if __name__ == '__main__':
    ## Create pickle session
    myseq = MySeq()

    myseq_pickled = PickeledSequence(pickles_dir="/tmp/", batches_per_pickle=10)
    myseq_pickled.save_pickles(myseq,
                               use_multiprocessing=True, shuffle=True, workers=4, max_queue_size=4, epochs=10,
                               pickle_size=3, compress=True)

    ## Train on pickled session
    # myseq_pickled = PickeledSequence(pickles_dir="pickles/", batches_per_pickle=10)
    # model.fit_generator(generator=myseq_pickled)
    #
