#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

echo "Building bbox_ad..."
cd ../mmdet/core/bbox
$PYTHON setup_linux.py build_ext --inplace

