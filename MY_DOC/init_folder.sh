#!/usr/bin/env bash
root="../"

ln -s ${root}../../data ${root}./data
ln -s ${root}../../data/mmdet_results/ ${root}./results
ln -s ${root}../../data/intermediate_results/ ${root}./intermediate_results
ln -s ${root}../../data/mmdet_checkpoints/ ${root}./checkpoints