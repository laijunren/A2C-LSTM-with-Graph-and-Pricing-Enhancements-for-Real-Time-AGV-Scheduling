from modules.dataloader import load_data
from modules.dataparse import normalize_features
from modules.model import ActorNetwork
from modules.trainer import train
from modules.tester import test

import torch
import torch.nn as nn
import torch.optim as optim

def main():
    # 加载数据
    feature_tensor, edge_index, edge_weight = load_data(
        "/home/aaa/my_code/hospital-main/GCN_LSTM/data/data/adjacency_matrix.csv", 
        "/home/aaa/my_code/hospital-main/GCN_LSTM/data/data/degree_matrix.csv", 
        "/home/aaa/my_code/hospital-main/GCN_LSTM/data/data/average_lift_waiting_time.csv"
    )
    
    # 数据预处理
    feature_tensor = normalize_features(feature_tensor)
    
    # 初始化模型
    model = ActorNetwork(in_channels=1, hidden_channels=16, lstm_hidden_size=32, out_size=9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # 训练模型
    train(model, optimizer, loss_fn, feature_tensor, edge_index, epochs=100)
    
    # 测试模型
    output = test(model, feature_tensor, edge_index)
    print("Test Output:", output)

if __name__ == "__main__":
    main()
