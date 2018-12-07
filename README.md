# DLProject
Repo for ML: Deep Learning Final Project,

# Requirements
  - pytorch -> for modeling
  - nltk -> for bleu scores
  - python3.7 -> python standard used to run

# Points of Interest:
data\\

 - modern, orginal, processed, split data
 - format: original|||modern
 - old stuff is sparknotes, non old stuff is enotes
 - split\\ -> train/dev/test split files (randomized per whole corpus)

src\\

 - Source code for project
 - preprocessing\\ -> scripts to process & split data
 - results\\ -> raw translation files
 - results\\results_verbose\\ -> pretty translation files
 - results\\results_verbose\\linked_results.txt -> Easy read file (check it out!)
 - has a separate readme on CLI args and how to run

proposal\\

 - proposal files, including bibtex, tex, & pdf
 - use make file to recompile pdf

interim\\

  - interim report files
  - use make file to recompile pdf

final\\

  - final report files
  - use make file to recompile pdf
