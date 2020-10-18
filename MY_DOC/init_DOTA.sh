#!/usr/bin/env bash
cd ..
cd DOTA_devkit
python setup.py build_ext --inplace
cd ..
bash compile.sh