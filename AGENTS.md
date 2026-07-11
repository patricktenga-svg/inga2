# AGENTS — AI assistant guidance for this repository

Purpose: provide concise, actionable instructions for AI coding agents working in this codebase so they can be productive immediately.

Quick start
- Run the app locally: `streamlit run app.py` (requires Python >=3.9 and dependencies in `requirements.txt`).
- Install dependencies: `pip install -r requirements.txt`.

Project layout (important files)
- `app.py`: Streamlit app entrypoint and main runtime loop.
- `requirements.txt`: Python dependencies.
- `data/historical_inflows.csv`: historical inflow data used for training.
- `models/`: model artifacts and training code (`lstm_model.py`, `classifier.py`, `anomaly_detector.py`, `rl_agent.py`).
- `utils/`: helpers for export and visualizations.

Agent conventions
- Prefer non-destructive edits and run tests or a quick local smoke run (e.g., `streamlit run app.py`) when changing runtime code.
- Do not commit large binary model files without checking repository policy; prefer storing model artifacts outside the repo and reference them by path.
- Link to existing documentation instead of duplicating large sections. If adding project-specific guidance, keep it short and focused.

Runtime and training notes
- The Streamlit UI uses caching decorators: `@st.cache_data` and `@st.cache_resource`. Respect these when refactoring.
- Models are saved/loaded with a `path_prefix` convention producing files like `models/inga_ii_lstm.pth`, `models/inga_ii_classifier.pkl`, `models/inga_ii_anomaly.pkl`, `models/inga_ii_actor.pth`.
- Training may use CPU-bound PyTorch and scikit-learn; long training should be performed outside the interactive Streamlit session when possible.

Common tasks for agents
- Small code changes and fixes: edit files directly and run `streamlit run app.py` to smoke-test.
- Add unit tests: repository currently has no tests — suggest adding a `tests/` folder and a simple `pytest` workflow.
- Add CI: suggest a lightweight GitHub Actions workflow to run `pip install -r requirements.txt` and `pytest`.

Where to look for further context
- Data processing and plotting: `utils/data_exporter.py`, `utils/visualizations.py`.
- Model training and APIs: `models/lstm_model.py`, `models/classifier.py`, `models/anomaly_detector.py`, `models/rl_agent.py`.

If you want me to create other agent customizations (instruction files, smaller skills, CI hooks), ask and specify the target (tests, CI, model management).
