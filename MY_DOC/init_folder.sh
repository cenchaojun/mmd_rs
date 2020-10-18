#!/usr/bin/env bash
cd ..
rm data
rm resutls
rm intermediate_results
rm checkpoints
ln -s ../../data ./data
ln -s ../../data/mmdet_results/ ./results
ln -s ../../data/intermediate_results/ ./intermediate_results
ln -s ../../data/mmdet_checkpoints/ ./checkpoints