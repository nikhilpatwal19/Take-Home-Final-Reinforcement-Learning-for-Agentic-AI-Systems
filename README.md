# Reinforcement Learning for Agentic AI Systems
## Adaptive Tutorial Agent with Q-Learning, REINFORCE, UCB & Thompson Sampling

A multi-agent adaptive tutorial system where **multiple agents learn through RL** to optimize teaching strategies for simulated students. The DifficultyAgent learns what to teach (Q-Learning/REINFORCE), the FeedbackAgent learns how to give feedback (Thompson Sampling bandit), and the system adapts through experience with dynamic routing, cross-session persistence, and production-ready deployment.

---

## Overview

Four RL approaches integrated into an agentic tutorial system:

1. **Q-Learning + UCB** (Value-Based + Exploration Strategy) — Tabular Q-table with Bellman updates and UCB exploration
2. **REINFORCE** (Policy Gradient) — Monte Carlo policy gradient with softmax policy and running-mean baseline
3. **Transfer Learning** (Knowledge Transfer) — Q-table warm-starting from source domain to target domain
4. **Thompson Sampling** (Multi-Agent RL) — FeedbackAgent learns optimal feedback strategy via Beta posterior sampling

### Multi-Agent Architecture

| Agent | Role | Learning Method | Communication |
|---|---|---|---|
| **Difficulty Agent** | Selects question difficulty | Q-Learning + UCB / REINFORCE | Receives constraints from ProgressAgent, sends override protests |
| **Feedback Agent** | Selects feedback strategy | **Thompson Sampling bandit** | Sends pattern alerts to ProgressAgent; learns from next-Q accuracy |
| **Progress Agent** | Tracks milestones, constraints | Rule-based (trend analysis) | Sends difficulty caps/floors to DifficultyAgent |
| **TutorController** | Dynamic pipeline orchestration | N/A | Routes all messages; challenge mode; conditional routing |

### Custom Tools

- **DifficultyCalibrator** — Extracts curriculum recommendations from trained Q-table, exports to JSON
- **CurriculumValidator** — Validates difficulty sequences against 5 pedagogical rules
- **StudentProfiler** — Classifies student types from behavioral features, generates radar visualizations

### Key Engineering Features

- **Dynamic Controller Routing:** Challenge mode auto-activates on boredom + high accuracy; conditional FeedbackAgent skip for early steps
- **SQLite Persistence:** Cross-session storage, analytics queries, feedback bandit statistics
- **Production Deployment:** FastAPI-ready TutorDeployment class with session management and historical querying

---

## Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Setup
```bash
git clone https://github.com/nikhilpatwal19/Take-Home-Final-Reinforcement-Learning-for-Agentic-AI-Systems.git
cd rl-adaptive-tutorial-agent
pip install -r requirements.txt
jupyter notebook RL_Adaptive_Tutorial_Agent.ipynb
```

### Dependencies
```
numpy
matplotlib
scipy
```
No GPU, API keys, or external services required. SQLite is included in Python's standard library.

---

## How to Run

1. Open `RL_Adaptive_Tutorial_Agent.ipynb` in Jupyter
2. Run all cells top to bottom (`Cell → Run All`)
3. Expected runtime: ~8–12 minutes
4. All figures saved as PNGs; JSON reports and SQLite database generated automatically

---

## Key Results

| Agent | Avg Reward | Avg Final Skill | vs Random |
|---|---|---|---|
| Random Baseline | -23.7 | 0.40 | — |
| Static Curriculum | -5.3 | 0.44 | +18.4 pts |
| ε-Greedy Q-Learning | 32.5 | 0.50 | +56.2 pts |
| **UCB Q-Learning** | **35.9** | **0.50** | **+59.6 pts** |
| REINFORCE | ~30-34 | ~0.49 | +54-58 pts |

- All trained agents significantly outperform baselines (p < 0.001, Cohen's d > 2.0)
- UCB outperforms ε-Greedy (p < 0.05)
- Thompson Sampling bandit learns differentiated feedback strategies per student state
- Transfer learning provides measurable jumpstart advantage

---

## Repository Structure

```
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── RL_Adaptive_Tutorial_Agent.ipynb    # Main notebook (run top to bottom)
├── Technical_Report.pdf                # Comprehensive technical report
├── calibration_report.json             # Curriculum recommendations
├── student_profiles.json               # Student profile analyses
├── demo_session.json                   # Sample deployment session
├── tutor_sessions.db                   # SQLite session database (generated)
├── learning_curves.png                 # Training curves (3 agents)
├── convergence_analysis.png            # Q-value convergence
├── optimal_retrain_qtrajectory.png     # Q-value trajectories
├── reinforce_policy.png                # REINFORCE action probabilities
├── feedback_bandit_analysis.png        # Thompson Sampling preferences
├── policy_heatmap.png                  # Greedy policy visualization
├── robustness_test.png                 # Cross-student-type evaluation
├── statistical_comparison.png          # 5-way statistical comparison
├── baseline_comparison.png             # Baseline bar charts
├── before_after_comparison.png         # Untrained vs trained demo
├── agent_communication_trace.png       # Inter-agent message flow
├── difficulty_calibration.png          # Per-student difficulty progressions
├── student_profiles_radar.png          # Student behavioral feature radars
├── transfer_learning.png               # Transfer vs from-scratch curves
└── hyperparameter_sensitivity.png      # Parameter sweep results
```

---

## Ethical Considerations

- **Alignment:** Reward optimizes for genuine learning, not engagement metrics
- **Bias:** Robustness tested across 4 student types; production requires demographic validation
- **Privacy:** SQLite stores only aggregated metrics locally; FERPA compliance required in production
- **Transparency:** Feedback strategies logged; students should know the system adapts to them
- **Fairness:** StudentProfiler classifications must not correlate with protected attributes

---

## License

This project was developed as a course assignment for Reinforcement Learning for Agentic AI Systems.
