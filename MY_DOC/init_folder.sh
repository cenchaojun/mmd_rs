#!/usr/bin/env bash
cd ..
rm data
rm results
rm intermediate_results
rm checkpoints
ln -s /opt/data/nfs/huangziyue ./data
ln -s /opt/data/nfs/huangziyue/mmdet_results/ ./results
ln -s /opt/data/nfs/huangziyue/intermediate_results/ ./intermediate_results
ln -s /opt/data/nfs/huangziyue/mmdet_checkpoints/ ./checkpoints
