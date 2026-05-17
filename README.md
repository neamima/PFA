# PFA Melanome IA

This repository contains an image-based melanoma diagnostic project combining Python backend model training/evaluation and a React frontend for inference and user interaction.

## Repository structure

- `backend/` — Python API, model training and utilities
	- `main.py` — backend entrypoint / API server
	- `requirements.txt` — Python dependencies
	- `models/` — pre-trained model weights (large files, gitignored)
	- `src/` — dataset, model, training and utils modules
- `frontend/` — React + Vite frontend for user interface
- `data/` — dataset folders (raw, processed). Keep large datasets out of Git.
- `notebooks/` — exploration and evaluation notebooks

## Quick start

Prerequisites:

- Python 3.9+ (recommended)
- Node.js 16+ and npm or yarn

Setup backend (from repo root):

```bash
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

Start backend API (example):

```bash
python backend/main.py
```

Setup and start frontend:

```bash
cd frontend
npm install
npm run dev
```

Open the app at the address printed by Vite (usually `http://localhost:5173`).

## Models & Data

- Pretrained weights are stored locally in `backend/models/`. These files are large and are ignored by Git by default.
- Place raw images and metadata under `data/raw/` and processed data under `data/processed/` before training or evaluation.
- Example model files included in `.gitignore`: `*.pth`, `*.pt`, `*.ckpt`.

## Training & Evaluation

- See `backend/src/train.py` for the training loop and hyperparameters.
- See `notebooks/01_exploration_et_pretraitement.ipynb` and `02_evaluation_comparative.ipynb` for data exploration and comparative evaluation.

## Development notes

- Virtual environments and editor folders are ignored (`venv/`, `.vscode/`).
- Keep large binary artifacts (models, datasets) out of Git; use external storage or Git LFS if needed.

## Contributing

1. Fork the repo
2. Create a branch `feature/your-feature`
3. Run tests and linters, open a PR

## License & Contact

Add your preferred license here. For questions, contact the project owner.

