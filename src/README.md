# Source Code

**Morris Kraicer (mkraice1@jhu.edu), Riley Scott (rscott39@jhu.edu), William Watson
(billwatson@jhu.edu)**

## Usage

Training:
```
$ python3 -O main.py [CLI ARGS]
```

Notable CLI arguments for model building:

```
  --hidden-size HIDDEN_SIZE
                        hidden size of encoder/decoder
  --embedding-dropout EMBEDDING_DROPOUT
                        training embedding dropout rate
  --rnn-dropout RNN_DROPOUT
                        training RNN dropout rate
  --encoder-layers ENCODER_LAYERS
                        number of RNN layers in the encoder
  --decoder-layers DECODER_LAYERS
                        number of RNN layers in the decoder
  --train-batch-size BATCH_SIZE
                        number of same length source sentences to batch
  --teacher-forcing-chance TEACHER_FORCING_CHANCE
                        percent of batches on which to teacher force
  --encoder-mode {rnn, gru, bigru}
                        Encoder type
  --decoder-mode {rnn, gru}
                        decoder type   
  --attention-mode {none, dot, concat}
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
loss.py - Extended loss function to handle masking
model.py - Encapsulates Encoder-Decoder model (+ train)
main.py - Main file
trainer.py - File to handle interactions with model (train, eval, translate)
utils.py - methods for timing, chunking, BLEU metrics
vocab.py - Builds vocab, handles token indexing
```

## Full CLI list:
CLI:
```
usage: main.py [-h] [--hidden-size HIDDEN_SIZE]
               [--embedding-dropout EMBEDDING_DROPOUT]
               [--rnn-dropout RNN_DROPOUT] [--encoder-layers ENCODER_LAYERS]
               [--decoder-layers DECODER_LAYERS]
               [--encoder-mode {rnn,gru,bigru}] [--decoder-mode {rnn,gru}]
               [--attention-mode {none,general,concat}]
               [--source-language SOURCE_LANGUAGE]
               [--target-language TARGET_LANGUAGE] [--train-file TRAIN_FILE]
               [--dev-file DEV_FILE] [--test-file TEST_FILE]
               [--train-batch-size TRAIN_BATCH_SIZE] [--reverse]
               [--max-length MAX_LENGTH] [--epochs EPOCHS]
               [--log-every LOG_EVERY] [--checkpoint-every CHECKPOINT_EVERY]
               [--checkpoint-prefix CHECKPOINT_PREFIX]
               [--initial-learning-rate INITIAL_LEARNING_RATE]
               [--teacher-forcing-chance TEACHER_FORCING_CHANCE]
               [--load-checkpoint LOAD_CHECKPOINT] [--use-gpu]
```

List of CLI args and usage:
```
optional arguments:
  -h, --help            show this help message and exit
  --hidden-size HIDDEN_SIZE
                        hidden size of encoder/decoder (word vector size)
  --embedding-dropout EMBEDDING_DROPOUT
                        training embedding dropout rate
  --rnn-dropout RNN_DROPOUT
                        training RNN/GRU dropout rate
  --encoder-layers ENCODER_LAYERS
                        number of RNN layers in the encoder
  --decoder-layers DECODER_LAYERS
                        number of RNN layers in the decoder
  --encoder-mode {rnn,gru,bigru}
                        type of encoder used
  --decoder-mode {rnn,gru}
                        type of decoder used
  --attention-mode {none,general,concat}
                        type of attention used
  --source-language SOURCE_LANGUAGE
                        source sentence language
  --target-language TARGET_LANGUAGE
                        target sentence language
  --train-file TRAIN_FILE
                        train sentences
  --dev-file DEV_FILE   dev sentences
  --test-file TEST_FILE
                        test sentences
  --train-batch-size TRAIN_BATCH_SIZE
                        sets the training batch size
  --reverse             if True reverses source and target reading
  --max-length MAX_LENGTH
                        max length of a decoded output sentence
  --epochs EPOCHS       total number of epochs to train on
  --log-every LOG_EVERY
                        log loss info every this many epochs
  --checkpoint-every CHECKPOINT_EVERY
                        write out checkpoint every this many epochs
  --checkpoint-prefix CHECKPOINT_PREFIX
                        checkpoint filename prefix
  --initial-learning-rate INITIAL_LEARNING_RATE
                        initial learning rate
  --teacher-forcing-chance TEACHER_FORCING_CHANCE
                        percent of batches on which to teacher force
  --load-checkpoint LOAD_CHECKPOINT
                        training checkpoint to load
  --use-gpu             use CPU/CUDA for training/evaluation

```
