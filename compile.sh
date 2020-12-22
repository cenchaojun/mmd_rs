#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

cd mmdet/ops/roi_align_rotated

echo "Building roi align rotated op..."
cd ../roi_align_rotated
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building ps roi align rotated op..."
cd ../psroi_align_rotated
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace


echo "Building poly_nms op..."
cd ../poly_nms
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building sigmoid focal loss op..."
cd ../sigmoid_focal_loss_old
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace


echo "Building cpu_nms..."
cd ../../core/bbox
$PYTHON setup_linux_ad.py build_ext --inplace

