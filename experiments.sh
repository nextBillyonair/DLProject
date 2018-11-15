#!/bin/bash

cd src


# 1
python3 main.py --encoder-mode rnn --decoder-mode rnn --attention-mode none \
        --use-gpu > results/rnn.txt
# 2
python3 main.py --encoder-mode gru --decoder-mode gru --attention-mode none \
        --use-gpu > results/gru.txt
# 3
python3 main.py --encoder-mode bigru --decoder-mode gru --attention-mode none \
        --use-gpu > results/bigru.txt
# 4
python3 main.py --encoder-mode bigru --decoder-mode gru \
        --attention-mode concat --use-gpu > results/bigru_concat.txt
# 5
python3 main.py --encoder-mode bigru --decoder-mode gru \
        --attention-mode dot --use-gpu > results/bigru_dot.txt
# 6
python3 main.py --encoder-mode bigru --decoder-mode gru \
        --attention-mode concat --teacher-forcing-chance 0.5 \
        --use-gpu > results/bigru_concat_50.txt
# 7
python3 main.py --encoder-mode bigru --decoder-mode gru \
        --attention-mode dot --teacher-forcing-chance 0.5 \
        --use-gpu > results/bigru_dot_50.txt
# 8
python3 main.py --encoder-mode bigru --decoder-mode gru \
        --attention-mode concat --teacher-forcing-chance 0.5 \
        --use-gpu > results/bigru_concat_100.txt
# 9
python3 main.py --encoder-mode bigru --decoder-mode gru \
        --attention-mode dot  --teacher-forcing-chance 0.5 \
        --use-gpu > results/bigru_dot_100.txt
