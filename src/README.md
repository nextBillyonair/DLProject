# Source Code

**Morris Kraicer (mkraice1@jhu.edu), Riley Scott (rscott39@jhu.edu), William Watson
(billwatson@jhu.edu)**

## Usage

Training:
```
$ python3 -O seq2seq.py --encoder-layers 4 --decoder-layers 4 \
    --train-batch-size 2048
```

Notable CLI arguments for model building:

```
  --hidden-size HIDDEN_SIZE
                        hidden size of encoder/decoder
  --embedding-dropout EMBEDDING_DROPOUT
                        training embedding dropout rate
  --lstm-dropout LSTM_DROPOUT
                        training LSTM dropout rate
  --encoder-layers ENCODER_LAYERS
                        number of LSTM layers in the encoder
  --decoder-layers DECODER_LAYERS
                        number of LSTM layers in the decoder
  --train-batch-size BATCH_SIZE
                        number of same length source sentences to batch
  --teacher-forcing-chance TEACHER_FORCING_CHANCE
                        percent of batches on which to teacher force
  --encoder-mode {baseline, gru, bidirectional}
                        Encoder type
  --decoder-mode {baseline, gru}
                        decoder type   
  --attention-mode {None, dot, concat}
                        attention type                   
```

Notable CLI arguments for training parameters:

```
  --train-batch-size TRAIN_BATCH_SIZE
                        number of samples in a training batch
  --epochs EPOCHS
                        number of training epochs
  --initial-learning-rate INITIAL_LEARNING_RATE
                        starting learning rate
```

Notable CLI arguments for checkpointing/logging:
```
  --log-every LOG_EVERY
                        log loss info every this many epochs
  --checkpoint-every CHECKPOINT_EVERY
                        Write out checkpoint every this many epochs
  --load-checkpoint LOAD_CHECKPOINT
                        training checkpoint to load
```

## Organization

We provide the file + uses:

```
args.py - CLI arguments
attention_models.py - Attention Layers
dataset.py - Handles vocab building + batching, etc for data
decoders.py - All Decoder Types
encoders.py - All Encoder Types
model.py - Encapsulates Encoder-Decoder model
seq2seq.py - Main file
trainer.py - File to handle interactions with model (train, eval, translate)
utils.py - methods for timing, chunking, BLEU metrics
vocab.py - Builds vocab, handles token indexing
```
