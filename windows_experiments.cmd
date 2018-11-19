cd src
python main.py --encoder-mode rnn --decoder-mode rnn --attention-mode none --use-gpu > results/rnn.txt
python main.py --encoder-mode gru --decoder-mode gru --attention-mode none --use-gpu > results/gru.txt
python main.py --encoder-mode bigru --decoder-mode gru --attention-mode none --use-gpu > results/bigru.txt
python main.py --encoder-mode bigru --decoder-mode gru --attention-mode concat --use-gpu > results/bigru_concat.txt
python main.py --encoder-mode bigru --decoder-mode gru --attention-mode dot --use-gpu > results/bigru_dot.txt
python main.py --encoder-mode bigru --decoder-mode gru --attention-mode concat --teacher-forcing-chance 0.5 --use-gpu > results/bigru_concat_50.txt
python main.py --encoder-mode bigru --decoder-mode gru --attention-mode dot --teacher-forcing-chance 0.5 --use-gpu > results/bigru_dot_50.txt
python main.py --encoder-mode bigru --decoder-mode gru --attention-mode concat --teacher-forcing-chance 0.5 --use-gpu > results/bigru_concat_100.txt
python main.py --encoder-mode bigru --decoder-mode gru --attention-mode dot  --teacher-forcing-chance 0.5 --use-gpu > results/bigru_dot_100.txt