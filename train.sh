#!/bin/bash
#
#SBATCH -p cpl
#SBATCH --time=7-0
#SBATCH --gres=gpu:titan-x:1
#SBATCH --mem=40G
#

python main.py
