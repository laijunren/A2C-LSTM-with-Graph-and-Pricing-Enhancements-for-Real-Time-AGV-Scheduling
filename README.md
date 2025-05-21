
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
├── A2C_LSTM_GCN.py                           # GCN-based A2C-LSTM model
├── A2C_LSTM_Dual-value-pricing.py            # Dual-pricing A2C-LSTM training script
├── Environment.py                            # DRL-AnyLogic communication interface
├── bonsalt.py                                # HTTP simulator communication (BonsaltEnv)
├── Pricing.py                                # LP-based dual-value pricing module
├── data_config.py                            # Node/graph construction from hospital instance
├── Utils.py                                  # Auxiliary functions
├── gym.py                                    # Simple gym environment (baseline)
└── Simulator/                                # External folder (AnyLogic simulation environment)
```

---

## 🧠 Models

### 1. A2C–LSTM–GCN

- Encodes hospital layout into node embeddings via GCN.
- Combines spatial and temporal features via LSTM.
- Learns AGV selection policy; task sequencing handled by heuristic.

### 2. A2C_LSTM_Dual-value-pricing.py

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

After training, the following files are automatically saved under `./A2C_output/`:

- `Actor_<timestamp>.pth`: Trained actor network parameters
- `Critic_<timestamp>.pth`: Trained critic network parameters
- `Actor_optim_<timestamp>.pth`: Optimizer state for actor
- `Critic_optim_<timestamp>.pth`: Optimizer state for critic
- `makespan_<timestamp>.txt`: Per-episode task completion times
- `rewards_<timestamp>.txt`: Cumulative rewards per episode
- `actor_loss_<timestamp>.txt`: Actor loss values per episode
- `critic_loss_<timestamp>.txt`: Critic loss values per episode

---



## 📧 Contact

For questions, please contact:
- 🧑‍🔬 scxjl8@nottingham.edu.cn

---

## 🔄 Loading Pretrained Models

To continue training from saved checkpoints, this project includes a utility to **automatically load pretrained models**:

```python
from your_script import load_best_checkpoint

actor_network = ActorNetwork(...)
value_network = ValueNetwork(...)
checkpoint_dir = "./A2C_output/pth"

load_best_checkpoint(actor_network, checkpoint_dir, "Actor")
load_best_checkpoint(value_network, checkpoint_dir, "Critic")
```

This function:
- Scans the given directory for `.pth` files that contain the keywords `"Actor"` or `"Critic"`.
- Measures parameter compatibility and loads only the matching layers.
- Supports flexible resumption from partially matching models.

If you prefer to **manually load a specific file**, you can also use:

```python
actor_network.load_state_dict(torch.load("A2C_output/Actor_xx_xx_xx_xx.pth"))
```

Make sure your model structure matches the saved weights.
