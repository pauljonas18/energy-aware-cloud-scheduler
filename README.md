# DQN-Guided NSGA-II with Honey-Badger Search for Energy-Aware DAG Scheduling 

This repository contains the implementation and final thesis report for my Senior Design Project (SDP) in Computer Science. It proposes a hybrid multi-objective scheduler for heterogeneous cloud clusters.

## Project Overview
Workflow scheduling on heterogeneous cloud clusters is an NP-hard problem requiring the simultaneous minimization of makespan, cluster energy consumption, and per-task waiting time. State-of-the-art schedulers often optimize a single objective, leading to SLA violations. 

This project solves this by decoupling operator selection from execution:
1. **NSGA-II Evaluator:** Maintains a Pareto-front archive of non-dominated schedules across generations.
2. **Deep Q-Network (DQN) Meta-Controller:** A reinforcement learning agent built with PyTorch that learns which search operator (from a pool of 7) best advances the Pareto front.
3. **Honey-Badger Optimizer (HBO) Executor:** A custom discrete execution engine that applies the DQN's chosen moves (Dig, Honey, Sniff, Jump, Swap).

## Dataset & Evaluation
The scheduler was evaluated using the **Alibaba Cluster Trace v2018** (100 GB dataset). 
* Tested on workloads ranging from 100 to 500 tasks on a heterogeneous pool of up to 200 processors.
* **Results:** Achieved a **38–44% makespan reduction** over 6 classical baselines (GA, PSO, GWO, EHO, standalone HBO, and standalone RL) with a **0.0% Deadline Mismatch Ratio** (strict SLA compliance).

## 🛠️ Tech Stack
* **Language:** Python 3.11
* **Deep Learning:** PyTorch (DQN Meta-controller implementation)
* **Data Processing:** Pandas, NumPy (Alibaba trace manipulation)
* **Graph Theory:** NetworkX (DAG generation and HEFT upward-rank computation)
* **Statistical Analysis:** SciPy (Wilcoxon rank-sum and Friedman tests)

## Repository Structure
* `scheduler.ipynb`: The complete Jupyter Notebook containing the data preprocessing, multiprocessor environment simulation, classical baselines, and the proposed DQN-NSGA-II-HBO architecture.
* `DQN_NSGAII_Cloud_Scheduling_Thesis.pdf`: The full academic thesis report detailing the mathematical formulation, reward shaping mechanics, and step-by-step statistical results.

## Key Architectural Highlight: The 6D State Vector
To avoid the "Curse of Dimensionality" and prevent overfitting, the DQN does not observe the raw 500-node DAG. Instead, it observes a compressed 6-dimensional continuous state vector representing the mathematical "health" of the Pareto front:
1. Normalized Generation Index
2. Mean Score (Normalized by SLA)
3. Minimum/Best Score (Normalized by SLA)
4. Absolute Standard Deviation (Normalized by SLA)
5. Relative Standard Deviation (Normalized by Current Max)
6. Normalized Stagnation Counter

graph TD
    %% Define Styles
    classDef input output fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef ai fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef loop fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef eval fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;

    %% Input & Init
    A[📦 Alibaba Cluster Trace v2018<br/>Ingest DAGs & Processor Data]:::input
    B[⚙️ HEFT Initialization<br/>Compute Upward Ranks & Warm-Start]:::input

    A --> B
    B --> C

    %% Generational Loop
    subgraph Generational Loop [NSGA-II Generational Loop: 150 Iterations]
        direction TB
        C[📊 1. State Observation<br/>Extract 6D Pareto Front Vector]:::ai
        D[🧠 2. DQN Meta-Controller<br/>PyTorch Predicts Q-values]:::ai
        E[🦡 3. HBO Executor<br/>Apply: Dig, Honey, Sniff, Jump...]:::loop
        F[⏱️ 4. Multi-Objective Eval<br/>Makespan, Energy, Wait Time]:::eval
        G[💰 5. Reward & Learning<br/>Δ Hypervolume & Bellman Update]:::ai
        H[⚖️ 6. Archive Maintenance<br/>NSGA-II Non-Dominated Sort]:::loop

        C --> D
        D --> E
        E --> F
        F --> G
        G --> H
    end

    %% Loop back and Output
    H -->|Next Generation| C
    H -->|Max Generations Reached| I

    %% Output
    I[🏆 Final Pareto Front<br/>Schedules with 0% Deadline Mismatch]:::input
