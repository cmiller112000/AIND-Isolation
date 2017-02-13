#!/bin/bash
for ((j=1; j<=10; j++)); do
    echo "Running Set: # $j"
    ts=$(date +"%Y%m%d_%H%M%S")
    for agent in game_agent.py_*
    do
        cp $agent game_agent.py
        for ((i=1; i<=10; i++)); do
            echo "Running: ${agent} - Run # $i"
            python ./tournament.py > tournament.out
            grep "\%" tournament.out >> results_${agent}_${ts}.out
        done
    done
done