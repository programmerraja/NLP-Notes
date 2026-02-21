
uv run -m spacy download en_core_web_lg
uv run -m spacy download en_core_web_trf


uv pip install pip

source .venv/bin/activate


 uv add presidio-analyzer presidio-anonymizer spacy spacy-transformers rapidfuzz