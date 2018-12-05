# Interim Report
Here are all the files that support the creation of our interim
report for DL

# Randomly Generating Sentences:
`sort -R ../data/split/train.snt.aligned | head -n N > output`

where N is the number of sentences to sample. Useful for observing
random N aligned sentences from train.

# Compile:
Use the makefile:
```
make
```
This compiles the report
