#!/bin/bash

submit_simulation () 
{
    debye=$1

    current_dir=$(pwd)

    jobname=pacsim_15mVclassical_debye${debye}
    
    mkdir "debye${debye}"
    cp run.yaml "debye${debye}"
    cp configuration.yaml "debye${debye}"
    cp lattice.lmp "debye${debye}"
    cp run_config.yaml "debye${debye}"
    cp run.sh "debye${debye}"
    cp processing.py "debye${debye}"
    cd "debye${debye}"
    sed -i '' "s/_DEBYE/${debye}/g" run.yaml


    bash run.sh
    cd - 
   
}

for debye in $(seq -f "%.1f" 3.0 3.0 24.0); do
    submit_simulation "$debye"
done
