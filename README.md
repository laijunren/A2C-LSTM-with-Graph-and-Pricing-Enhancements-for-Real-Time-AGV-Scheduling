# Hybrid A2C–LSTM with Dual-Pricing Enhancement for Real-Time Hospital AGV Routing

This repository implements a hybrid deep reinforcement learning (DRL) framework for real-time dispatching of automated guided vehicles (AGVs) in multi-floor hospitals. The system integrates:

- **A2C–LSTM** architecture for sequential policy learning
- **Graph Convolutional Networks (GCN)** to capture spatial structure
- **Dual-Value Pricing signals** extracted from LP relaxations to enhance congestion sensitivity

Simulations run in a high-fidelity **AnyLogic-based digital twin**, capturing elevator delays, stochastic arrivals, and energy constraints.

---

## 🚑 Use Case

Efficiently dispatching AGVs in hospitals is challenging due to:
- Dynamic task arrival
- Multi-floor navigation with elevator bottlenecks
- Charging constraints and floor-specific access

This framework provides adaptive policies that minimize makespan under such constraints.

---

## 📁 Project Structure

```
.
├── A2C_LSTM.py                          # Baseline actor-critic model
├── A2C_LSTM_Dual-value-pricing.py      # Pricing-based enhancement model
├── A2C_LSTM_GCN.py                      # GCN-based enhancement model
├── Environment.py                       # Python-AnyLogic environment wrapper
├── Pricing.py                           # LP relaxation and dual variable extraction
├── bonsalt.py                           # REST API simulator interface (Bonsalt)
├── data_config.py                       # Graph and feature preprocessing
├── Utils.py                             # Reward functions and helpers
├── gym.py                               # Gym baseline variant (for testing)
├── requirements.txt                     # Python dependencies
└── Simulator/                           # AnyLogic model scripts
```

---

## 🔧 Setup Instructions

### Environment

```bash
conda create -n agv-rl python=3.8
conda activate agv-rl
pip install -r requirements.txt
```

> 💡 Use the appropriate PyTorch Geometric wheel from https://pytorch-geometric.com/ based on your CUDA version.

### Start Simulation Services

```bash
./bonsalt_release_1_1_linux/Bonsalt.Daemon
./Simulator/Hospital_DRL/start-simulator.sh
```

---

## 🧠 Model Training

```bash
# Baseline
python A2C_LSTM.py

# Dual-pricing enhanced
python A2C_LSTM_Dual-value-pricing.py

# GCN-enhanced
python A2C_LSTM_GCN.py
```

Training logs and weights are stored in `A2C_output/`.

---

## 🧮 Dual-Value Pricing

Located in `Pricing.py`, this module:

- Solves a relaxed Set-Covering Problem using Gurobi
- Extracts marginal values (dual variables)
- Injects them into the state and reward for resource-aware scheduling

---

## 🧬 GCN Representation

In `A2C_LSTM_GCN.py`:

- Uses `GATConv` layers (PyG) over spatial node graphs
- Learns task embeddings and fuses them with temporal state via LSTM
- Enhances convergence and elevator bottleneck adaptation

---

## 📈 Output Files

Under `A2C_output/`:

- `Actor_*.pth`, `Critic_*.pth`: Model weights
- `*_optim_*.pth`: Optimizer states
- `makespan_*.txt`, `rewards_*.txt`: Episode metrics
- `actor_loss_*.txt`, `critic_loss_*.txt`: Training losses

---

## 🔄 Load Pretrained Models

```python
from Utils import load_best_checkpoint
load_best_checkpoint(model, "A2C_output/pth", "Actor")
```

You may also manually load:
```python
model.load_state_dict(torch.load("A2C_output/Actor_XX.pth"))
```

---

## 📊 Dataset

- 30 synthetic task instances
- Three scales: Small (20–95), Medium (116–200), Large (256–328)
- Includes floor-aware task graphs and LP dual-value labels

---
