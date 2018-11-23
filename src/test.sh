#!/bin/sh

python3 -m unittest discover -s . -p '*_test.py'
# python3 -m unittest discover -s . -p '*_test.py' 2>&1 | tee test_result.txt
