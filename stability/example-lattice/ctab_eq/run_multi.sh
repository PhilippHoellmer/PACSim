#!/bin/bash

submit_simulation () 
{
    charge=$1

    current_dir=$(pwd)

    jobname=pacsim_15mVclassical_charge${charge}
    
    mkdir "charge${charge}"
    cp run.yaml "charge${charge}"
    cp configuration.yaml "charge${charge}"
    cp lattice.lmp "charge${charge}"
    cp run_config.yaml "charge${charge}"
    cp run.sh "charge${charge}"
    cp processing.py "charge${charge}"
	cp run_simulation.sh "charge${charge}"
    cd "charge${charge}"
    sed -i '' "s/_CHARGE/${charge}/g" configuration.yaml


    bash run_simulation.sh
    cd - 
   
}

for charge in $(seq -f "%.1f" -5.0 -10.0 -45.0); do
    submit_simulation "$charge"
done
