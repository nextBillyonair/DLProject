#!/bin/bash

printf 'RNN\n'
python3 -W ignore eval.py --eval-file rnn.txt --stats-only

printf '\nGRU\n'
python3 -W ignore eval.py --eval-file gru.txt --stats-only

printf '\nBIGRU\n'
python3 -W ignore eval.py --eval-file bigru.txt --stats-only

printf '\nBIGRU+CONCAT\n'
python3 -W ignore eval.py --eval-file bigru_concat.txt --stats-only

printf '\nBIGRU+CONCAT+50TF\n'
python3 -W ignore eval.py --eval-file bigru_concat_50.txt --stats-only

printf '\nBIGRU+CONCAT+100TF\n'
python3 -W ignore eval.py --eval-file bigru_concat_100.txt --stats-only

printf '\nBIGRU+GENERAL\n'
python3 -W ignore eval.py --eval-file bigru_general.txt --stats-only

printf '\nBIGRU+GENERAL+50TF\n'
python3 -W ignore eval.py --eval-file bigru_general_50.txt --stats-only

printf '\nBIGRU+GENERAL+100TF\n'
python3 -W ignore eval.py --eval-file bigru_general_100.txt --stats-only

printf '\nBEST_FILE\n'
python3 -W ignore eval.py --eval-file best_file.txt --stats-only
