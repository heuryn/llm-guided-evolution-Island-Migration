#!/bin/bash
rm -rf first_test/* 0/* slurm* Report* 
rm -rf sota/ExquisiteNetV2/models/* sota/ExquisiteNetV2/weight/* sota/ExquisiteNetV2/results/*

sbatch master.sbatch