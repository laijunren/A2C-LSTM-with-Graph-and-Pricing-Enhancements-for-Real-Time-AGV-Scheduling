#!/bin/bash

# 切换到工作目录并激活 Conda 环境
echo "Changing directory and activating Conda environment..."
cd /home/aaa/my_code/hospital-main/
source activate py310

# 启动 tmux 会话
echo "Starting tmux session..."
tmux new-session -d -s hospital

# 终端 1: 启动 Bonsalt Daemon
echo "Starting Bonsalt Daemon..."
tmux new-window -t hospital:1 -n "Bonsalt Daemon" "./bonsalt_release_1_1_linux/Bonsalt.Daemon"

# 终端 2: 启动 Hospital Simulator
echo "Starting Hospital Simulator..."
tmux new-window -t hospital:2 -n "Hospital Simulator" "./Simulator/Hospital_DRL/start-simulator.sh"

# 终端 3: 运行 A2C_LSTM 训练
echo "Running A2C_LSTM training..."
tmux new-window -t hospital:3 -n "A2C_LSTM Training" "python code_RL/A2C_LSTM_train.py"

# 切换到 tmux 会话
echo "Switching to tmux session..."
tmux attach-session -t hospital