#!/bin/bash

for ((i=1; i<=10; i++)); do
    python ./tournament.py > tournament.out
    grep "\%" tournament.out >> results.out
done