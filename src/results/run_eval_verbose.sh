#!/bin/bash

printf 'RNN\n'
python3 -W ignore eval.py --eval-file rnn.txt > results_verbose/rnn.txt

printf '\nGRU\n'
python3 -W ignore eval.py --eval-file gru.txt > results_verbose/gru.txt

printf '\nBIGRU\n'
python3 -W ignore eval.py --eval-file bigru.txt > results_verbose/bigru.txt

printf '\nBIGRU+CONCAT\n'
python3 -W ignore eval.py --eval-file bigru_concat.txt > results_verbose/bigru_concat.txt

printf '\nBIGRU+CONCAT+50TF\n'
python3 -W ignore eval.py --eval-file bigru_concat_50.txt > results_verbose/bigru_concat_50.txt

printf '\nBIGRU+CONCAT+100TF\n'
python3 -W ignore eval.py --eval-file bigru_concat_100.txt > results_verbose/bigru_concat_100.txt

printf '\nBIGRU+GENERAL\n'
python3 -W ignore eval.py --eval-file bigru_general.txt > results_verbose/bigru_general.txt

printf '\nBIGRU+GENERAL+50TF\n'
python3 -W ignore eval.py --eval-file bigru_general_50.txt > results_verbose/bigru_general_50.txt

printf '\nBIGRU+GENERAL+100TF\n'
python3 -W ignore eval.py --eval-file bigru_general_100.txt > results_verbose/bigru_general_10.txt

printf '\nBEST_FILE\n'
python3 -W ignore eval.py --eval-file best_file.txt > results_verbose/best_file.txt
