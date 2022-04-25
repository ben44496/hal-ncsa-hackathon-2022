#!/bin/bash
horovodrun -np 4 python train.py --batch-size 5

# Change batch size and np to 4
