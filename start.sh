#!/bin/bash

echo "Changing directory and activating Conda environment..."
cd /home/aaa/my_code/hospital-main/
source activate py310

echo "Starting tmux session..."
tmux new-session -d -s hospital

echo "Starting Bonsalt Daemon..."
tmux new-window -t hospital:1 -n "Bonsalt Daemon" "./bonsalt_release_1_1_linux/Bonsalt.Daemon"

echo "Starting Hospital Simulator..."
tmux new-window -t hospital:2 -n "Hospital Simulator" "./Simulator/Hospital_DRL/start-simulator.sh"

echo "Running A2C_LSTM training..."
tmux new-window -t hospital:3 -n "A2C_LSTM Training" "python code_RL/A2C_LSTM_train.py"

echo "Switching to tmux session..."
tmux attach-session -t hospital
