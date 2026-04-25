"""Backward-compatible constant config based on central typed settings."""

from config.settings import get_settings

_settings = get_settings()

# LLM Configuration
LLM_MODEL = _settings.llm_model
OLLAMA_BASE_URL = _settings.ollama_base_url
EMBEDDING_MODEL = _settings.embedding_model

# Chroma Configuration
PERSIST_DIRECTORY = str(_settings.persist_directory_path)
KNOWLEDGE_BASE_PATH = str(_settings.knowledge_base_file_path)
RETRIEVER_TOP_K = _settings.retriever_top_k

# Generation Configuration
TEMPERATURE = _settings.temperature
MAX_TOKENS = _settings.max_tokens

# DeepEval Configuration
DEEPEVAL_MODEL = _settings.deepeval_model
DEEPEVAL_API_KEY = _settings.deepeval_api_key

# RAGAS Configuration
RAGAS_MODEL = _settings.ragas_model
RAGAS_API_KEY = _settings.ragas_api_key
