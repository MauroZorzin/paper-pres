python -m venv .venv

source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install --upgrade openai tqdm

export OPENAI_API_KEY="sk-..."   # or set via your shell / env manager
