
# Hybrid A2C–LSTM for Real-Time Hospital AGV Scheduling

This repository implements a hybrid deep reinforcement learning framework for real-time AGV (Automated Guided Vehicle) dispatching in multi-floor hospital environments. The method integrates an LSTM-enhanced Actor-Critic policy with two key enhancements:

- **Graph Convolutional Networks (GCN)** for modeling spatial connectivity (e.g., elevators, charging points).
- **Dual-Value Pricing Signals** derived from linear programming (LP) relaxations to improve congestion awareness.

The system is trained and evaluated within a **realistic AnyLogic-based hospital simulator**, supporting stochastic arrivals, elevator delays, and energy constraints.

## 🚑 Use Case

Hospitals face dynamic task arrivals, elevator congestion, and floor-specific routing constraints. This hybrid DRL system optimizes AGV makespan by learning efficient assignment policies under realistic constraints.

---

## 📁 Project Structure

```
.
├── A2C_LSTM_GCN.py              # GCN-based A2C-LSTM model
├── A2C_LSTM_train.py            # Dual-pricing A2C-LSTM training script
├── Environment.py               # DRL-AnyLogic communication interface
├── bonsalt.py                   # HTTP simulator communication (BonsaltEnv)
├── Pricing.py                   # LP-based dual-value pricing module
├── data_config.py               # Node/graph construction from hospital instance
├── Utils.py                     # Auxiliary functions
├── gym.py                       # Simple gym environment (baseline)
└── Simulator/                   # External folder (AnyLogic simulation environment)
```

---

## 🧠 Models

### 1. A2C–LSTM–GCN

- Encodes hospital layout into node embeddings via GCN.
- Combines spatial and temporal features via LSTM.
- Learns AGV selection policy; task sequencing handled by heuristic.

### 2. A2C–LSTM–Pricing

- Uses dual-values from LP-set-covering relaxation as pricing signals.
- Pricing integrated into both state vector and reward shaping.
- Enhances resource sensitivity under elevator congestion.

---

## 🧪 Training Instructions

### 1. Prerequisites

- Python 3.8+
- PyTorch ≥ 1.10
- PyTorch Geometric
- Gurobi Optimizer (academic license required)
- AnyLogic simulator (external)

### 2. Start the Simulator

Before launching the training, ensure the simulation services are running:

```bash
# Start the Bonsalt daemon (custom executable)
./bonsalt_release_1_1_linux/Bonsalt.Daemon

# Then launch the AnyLogic simulation
./Simulator/Hospital_DRL/start-simulator.sh
```

### 3. Train a Model

**A2C–LSTM baseline:**
```bash
python A2C_LSTM.py
```

**Dual-value pricing variant:**
```bash
python A2C_LSTM_Dual-value-pricing.py
```

**GCN-enhanced variant:**
```bash
python A2C_LSTM_GCN.py
```

Training logs, models, and reward curves are saved under `./A2C_output/`.

---

## 🔌 Integration with AnyLogic

The communication between Python and AnyLogic is handled via `bonsalt.py`, which wraps HTTP requests to a RESTful Bonsalt simulator API.

- **Reset episode:** triggers AnyLogic to generate a new simulation instance.
- **Step(action):** sends selected AGV ID and receives the next environment state and reward.

> Ensure the AnyLogic experiment is configured to run in server mode and listen on port `5000`.

---

## 🧮 Dual-Value Pricing Module

Located in `Pricing.py`, this module:

- Solves a relaxed Set-Covering Problem (SCP) using Gurobi.
- Extracts dual variables (marginal costs of node inclusion).
- Injects them into the state and reward space to guide congestion-sensitive learning.

---

## 📊 Output

- `makespan_*.txt`: per-episode task completion time
- `rewards_*.txt`: per-episode cumulative reward
- `actor_loss_*.txt`: actor network loss over training
- `critic_loss_*.txt`: critic network loss over training

---

## 📚 Citation

For academic usage, please cite our NeurIPS 2025 paper:

> Hybrid A2C–LSTM with Dual-Pricing Enhancement for Real-Time Hospital AGV Routing, NeurIPS 2025 (under review)

---

## 📧 Contact

For questions, please contact:
- 🧑‍🔬 MUYU LAI @ University of Nottingham Ningbo
