#!/bin/bash
source /home/dany_leguy/miniconda/bin/activate jupyter
conda env update -f ./environment.yaml
# rm condaenv.*.txt