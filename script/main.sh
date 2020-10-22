#!/bin/sh
nohup python main.py > result/DTA.log 2> result/DTA.err & 
echo $! > PID

