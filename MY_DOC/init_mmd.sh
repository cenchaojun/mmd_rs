#!/usr/bin/env bash
cd ..
pip install mmpycocotools
pip install -r requirements/build.txt
python setup.py develop