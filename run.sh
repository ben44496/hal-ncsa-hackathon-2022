#!/bin/bash
horovodrun -np 4 python Unet.py

# Change batch size and np to 4