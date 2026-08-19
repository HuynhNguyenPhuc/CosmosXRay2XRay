# Documentation Suite Index — CosmosXRay2XRay

Welcome to the comprehensive documentation suite for **CosmosXRay2XRay**. This index serves as a central navigation router across all technical, architectural, experimental, and mathematical documentation files.

---

## 🗺️ Navigation Matrix

| Document | Topic & Focus Area | Key Content |
| :--- | :--- | :--- |
| **[STRUCTURE.md](./STRUCTURE.md)** | Repository Map & Specifications | Visual directory tree, module responsibilities, and file specifications. |
| **[WALKTHROUGH.md](./WALKTHROUGH.md)** | Execution & Pipeline Walkthrough | Step-by-step instructions for data caching, Predict 2.5 training, and Transfer 2.5 training. |
| **[DATASET.md](./DATASET.md)** | Dataset Specs & Preprocessing | Details on NSCLC, MELA2022, TCIA, MosMed, VinDr, CT HU windowing, and `.npy` cache formatting. |
| **[RENDERER.md](./RENDERER.md)** | Volume Rendering & Camera Setup | 7-view camera extrinsics/intrinsics, absorption-emission raymarching physics, and PyTorch3D implementation. |
| **[BENCHMARK.md](./BENCHMARK.md)** | Evaluation Protocol & Metrics | Quantitative benchmarking for Predict 2.5 vs Transfer 2.5 (Edge vs Depth ControlNet). |
| **[EXECUTION.md](./EXECUTION.md)** | Master Retraining & Scaling Guide | Multi-GPU DDP training, precision (bf16-mixed), WandB/TensorBoard logging, and checkpoint conversion. |
| **[LITERATURE.md](./LITERATURE.md)** | Academic Context & Related Work | Review of Cosmos 2.5 video diffusion models, ControlNet conditioning, and 2D-to-3D X-ray synthesis. |
| **[GOTCHAS.md](./GOTCHAS.md)** | Technical & Physical Pitfalls | Submodule CUDA builds, VAE latent frame constraints, DDP EMA sync, and memory optimization. |
| **[PROPOSAL.md](./PROPOSAL.md)** | Project Roadmap & Milestones | Strategic goals, deliverables, and conference submission targets. |
| **[REVIEWS.md](./REVIEWS.md)** | Quality Assurance & Peer Review | Submission readiness checklists, ablation criteria, and review guidelines. |
| **[CONDITIONING.md](./CONDITIONING.md)** | Conditioning Ports by Backbone | Which signal goes through which port for Predict 2.5 / Transfer 2.5 / Cosmos 3; Plücker rays vs. `camera_pose` actions; reasoner-freeze policy. |
| **[cosmos-predict3/PLAN.md](./cosmos-predict3/PLAN.md)** | Cosmos 3 Migration & `predict3/` Scaffolding | Architecture deltas vs. Predict 2.5/Transfer 2.5, real backbone verification, native I2V design, separate-venv requirement, what's not done yet. |

---

## 🚀 Quick Navigation

- **For AI Agents & Developers**: See [`CLAUDE.md`](../CLAUDE.md) and [`AGENTS.md`](../AGENTS.md).
- **For Experiment Tracking**: See [`research-state.yaml`](../research-state.yaml) and [`research-log.md`](../research-log.md).
- **For Web UI & Inference**: Run `python app.py --share`.
