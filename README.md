PickeledSequence allows you to prepare data for training before hand and run training after on another machine.

What can I say... It is Amazing.

## Create pickle session

```
    myseq_pickled = PickeledSequence(pickles_dir="/tmp/", batches_per_pickle=10)
    myseq_pickled.save_pickles(myseq,
                               use_multiprocessing=True, shuffle=True, workers=4, max_queue_size=4, epochs=10,
                               pickle_size=3, compress=True)
```

## Train on pickled session
```
    myseq_pickled = PickeledSequence(pickles_dir="pickles/", batches_per_pickle=10)
    model.fit_generator(generator=myseq_pickled)
```
