#!/bin/bash
cd ./training
python ./train.py \
	--outputDirectory="./output_train" \
	--randomSeed=0 \
	--initialProfile="./heated_segments.csv" \
	--architecture="ResidualNet_2_6_24_1" \
	--duration=10.0 \
	--alpha=0.0001 \
	--scheduleFilepath="./schedule.csv" \
	--numberOfBoundaryPoints=256 \
	--numberOfDiffEquResPoints=256 \
	--displayResults
	