#!/bin/bash
horovodrun -np 4 python Unet.py --batch-size 3

# Change batch size and np to 4