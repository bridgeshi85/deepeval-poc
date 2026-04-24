"""Configuration for LLM and evaluation settings."""
import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Chroma Configuration
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "./chroma_db")

# DeepEval Configuration
DEEPEVAL_MODEL = os.getenv("DEEPEVAL_MODEL", "gpt-4")
DEEPEVAL_API_KEY = os.getenv("DEEPEVAL_API_KEY")

# RAGAS Configuration
RAGAS_MODEL = os.getenv("RAGAS_MODEL", "gpt-4")
RAGAS_API_KEY = os.getenv("RAGAS_API_KEY")
