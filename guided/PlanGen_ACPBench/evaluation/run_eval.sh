#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 -m evaluation.evaluate_atlas "$@"
