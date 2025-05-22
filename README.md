# Hybrid A2C–LSTM with Dual-Pricing Enhancement for Real-Time Hospital AGV Routing

This repository implements a hybrid deep reinforcement learning (DRL) framework for real-time dispatching of Automated Guided Vehicles (AGVs) in multi-floor hospitals. The system integrates:

- **A2C–LSTM** architecture for sequential policy learning
- **Graph Convolutional Networks (GCN)** to capture spatial structure (e.g., elevators, charging points)
- **Dual-Value Pricing signals** extracted from LP relaxations to enhance congestion sensitivity

Simulations run in a high-fidelity **AnyLogic-based digital twin**, capturing elevator delays, stochastic arrivals, and energy constraints.

---

## 🚑 Use Case

Efficiently dispatching AGVs in hospitals is challenging due to:
- Dynamic task arrivals
- Multi-floor navigation with elevator bottlenecks
- Charging constraints and floor-specific access

This framework provides adaptive policies that minimize makespan under these constraints.

---

## 📁 Project Structure

```

.
├── A2C\_LSTM.py                          # Baseline A2C–LSTM model
├── A2C\_LSTM\_Dual-value-pricing.py      # Pricing-based enhancement model
├── A2C\_LSTM\_GCN.py                      # GCN-based enhancement model
├── Environment.py                       # Python-AnyLogic environment wrapper
├── Pricing.py                           # LP relaxation and dual variable extraction
├── bonsalt.py                           # REST API simulator interface (Bonsalt)
├── data\_config.py                       # Graph and feature preprocessing
├── Utils.py                             # Reward functions and helpers
├── gym.py                               # Gym baseline variant (for testing)
├── requirements.txt                     # Python dependencies
└── Simulator/                           # AnyLogic model scripts

````

---

## 🔧 Setup Instructions

### Environment Setup

To set up your environment, run the following commands:

```bash
conda create -n agv-rl python=3.8
conda activate agv-rl
pip install -r requirements.txt
````

> 💡 If using **PyTorch Geometric**, use the appropriate wheel from [PyTorch Geometric](https://pytorch-geometric.com/) based on your CUDA version.

### Start Simulation Services

Before starting training, ensure the simulator services are running:

```bash
# Start Bonsalt Daemon (custom executable)
./bonsalt_release_1_1_linux/Bonsalt.Daemon

# Start the AnyLogic simulation
./Simulator/Hospital_DRL/start-simulator.sh
```

---

## 📥 AnyLogic Simulator Installation
To run the hospital simulation, you need to install AnyLogic. Here’s how:

1. Download AnyLogic
Go to the official AnyLogic website.

Choose the "Academic" or "Trial" version depending on your license type.

Download and install the software for your operating system (Windows, Mac, Linux).

2. Install AnyLogic
Follow the installation instructions specific to your OS:

Windows: Run the .exe file and follow the installation wizard.

Mac: Mount the .dmg file and drag the AnyLogic app to the Applications folder.

Linux: Follow the Linux installation guide for your distribution.

3. Configure AnyLogic for Server Mode
Once installed, configure AnyLogic to run in server mode to allow interaction with the simulator:

Open AnyLogic and create or open the hospital simulation model.

Ensure the model is running in server mode (configured in the simulation settings).

The simulator should listen on port 5000 for incoming connections from Python (via Bonsalt).


## 🧳 Instance Conversion Steps

To use different task instances for training, follow these steps to replace the `model.jar` file:

1. **Open the Instance Folder**: First, open the `instance` folder and select the required instance files.
2. **Open the Selected Instance File**: Open the selected instance file and locate the `model.jar` file inside it.
3. **Replace the model.jar File**: Navigate to the `Simulator/Hospital_DRL/` folder and replace the existing `model.jar` file with the one you selected in the previous step.
4. **Complete the Replacement**: After following these steps, the task instance (instance) will be successfully converted, and you can proceed with training.

This allows you to easily switch between different task instances for simulation and training.

---

## 🧠 Model Training

To train the models, run the respective scripts:

```bash
# Baseline A2C–LSTM model
python A2C_LSTM.py

# Dual-value pricing enhanced model
python A2C_LSTM_Dual-value-pricing.py

# GCN-enhanced A2C–LSTM model
python A2C_LSTM_GCN.py
```

Training logs and weights will be saved under `A2C_output/`.

---

## 🧮 Dual-Value Pricing

The **Pricing.py** module:

* Solves a relaxed Set-Covering Problem (SCP) using **Gurobi**.
* Extracts marginal values (dual variables), which represent the costs of including nodes.
* These dual values are integrated into both the state representation and reward shaping to improve congestion-aware learning.

---

## 🧬 GCN Representation

In `A2C_LSTM_GCN.py`, the model:

* Uses **GATConv** layers (from PyTorch Geometric) over a graph representing the hospital layout.
* Learns node embeddings that capture spatial and temporal features.
* The spatial features are fused with temporal state information using an LSTM to form a joint policy.

This model enhances convergence speed and adapts better to elevator congestion scenarios.

---

## 📈 Output Files

After training, the following files will be saved under `A2C_output/`:

* `Actor_*.pth`, `Critic_*.pth`: Model weights for the actor and critic networks
* `*_optim_*.pth`: Optimizer checkpoints for the actor and critic
* `makespan_*.txt`: Task completion time (makespan) for each episode
* `rewards_*.txt`: Cumulative rewards for each episode
* `actor_loss_*.txt`, `critic_loss_*.txt`: Loss history during training for the actor and critic networks

---

## 🔄 Load Pretrained Models

To load pretrained models and continue training, you can use the following function:

```python
from Utils import load_best_checkpoint
load_best_checkpoint(model, 'A2C_output/pth', 'Actor')
load_best_checkpoint(value_network, 'A2C_output/pth', 'Critic')
```

Alternatively, you can manually load a specific checkpoint:

```python
model.load_state_dict(torch.load("A2C_output/Actor_XX.pth"))
```

---

## 📊 Dataset

The dataset includes:

* 30 synthetic task instances
* Three task scales: **Small** (20–95 tasks), **Medium** (116–200 tasks), and **Large** (256–328 tasks)
* Task data includes: node types (pickup, delivery, charging, elevator), time windows, and floor layout
* LP dual-value pricing labels for reward shaping

---
