# 🤖 Collaborative Visual Localization for Modular Self-Reconfigurable Robots (MSRR)

Vision-based fault-tolerant collaborative localization for **infrastructure-free relative pose estimation** in dispersed MSRR systems (SnailBot).

---

## 🧠 Overview

Dispersed MSRR localization is difficult due to limited sensing/compute, occlusions, intermittent observations, and sensor faults.  
We provide a collaborative visual localization framework with fault-tolerant fusion validated in simulation, indoor tests, and outdoor field trials.

---

## ✨ Key Features

- Monocular camera + **ArUco marker arrays** for inter-module relative pose estimation
- Learning-based **fault detection & isolation** for spurious visual measurements
- Robust to spikes, low-/high-frequency glitches, and observation interruptions
- Decentralized, peer-to-peer data sharing and fusion

---

## 📁 Project Structure

```text
msrr_localization/
├── perception/          # ArUco detection & pose estimation
├── fdi_module/          # Fault detection & isolation (CNN + filters)
├── localization/        # Cooperative localization backend
├── communication/       # Inter-module messaging
├── simulation/          # Simulation experiments
├── experiments/         # Real-world experiment configs/scripts
└── docs/
```

## 🗓️ Release Timeline

### ✅ v0.1 — Perception MVP (2026.1)

-  `perception/` ArUco detection + camera calibration utils
-  `perception/` Relative pose estimation + covariance output

### 🛡️ v0.2 — Fault Detection & Isolation (2026.3)

-  `fdi_module/` Temporal window construction (local + relative states)
-  `fdi_module/` CNN inference API (confidence + error type)
-  `fdi_module/` Spike / low-frequency / high-frequency handling pipeline
-  Log format + replay script for offline evaluation

### 🔄 v0.3 — Collaborative Localization Backend (2026.5)

-  `localization/` Belief prediction + relative fusion (confidence-weighted)
-  `localization/` State alignment + covariance recalculation
-  `communication/` Standardized message schemas (pose, covariance, confidence)

### 🧪 v0.4 — Simulation & Benchmarks (2026.6)

-  `simulation/` Obstacle-controlled constraint scenarios + fault injection
-  Metrics: average error / RMSE + plotting scripts
-  Baseline interfaces for comparison

### 📦 v1.0 — Reproducible Experiments Release (2026.7)

-  `experiments/` Indoor pipeline configs + scripts
-  `experiments/` Outdoor/field pipeline data IO + trajectory export
-  `docs/` Setup, calibration, running guide, and reproducibility checklist
