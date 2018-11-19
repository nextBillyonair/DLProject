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
        --attention-mode general --use-gpu > results/bigru_general.txt
# 6
python3 main.py --encoder-mode bigru --decoder-mode gru \
        --attention-mode concat --teacher-forcing-chance 0.5 \
        --use-gpu > results/bigru_concat_50.txt
# 7
python3 main.py --encoder-mode bigru --decoder-mode gru \
        --attention-mode general --teacher-forcing-chance 0.5 \
        --use-gpu > results/bigru_general_50.txt
# 8
python3 main.py --encoder-mode bigru --decoder-mode gru \
        --attention-mode concat --teacher-forcing-chance 0.5 \
        --use-gpu > results/bigru_concat_100.txt
# 9
python3 main.py --encoder-mode bigru --decoder-mode gru \
        --attention-mode general  --teacher-forcing-chance 0.5 \
        --use-gpu > results/bigru_general_100.txt
