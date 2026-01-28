"""
Centralized Configuration
"""
import os

# --- Scraping ---
BASE_URL = "https://tombankhadi.com/"
DOMAIN = "tombankhadi.com"
MAX_PAGES = 100
PRIORITY_URLS = [
    "https://tombankhadi.com/",
    "https://tombankhadi.com/hadi-gold",
    "https://tombankhadi.com/kartlar",
    "https://tombankhadi.com/krediler",
    "https://tombankhadi.com/sikca-sorulan-sorular"
]

# --- Paths ---
# Use absolute or relative paths from project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, "rag_results")

# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "output.json")
BACKUP_FILE = os.path.join(OUTPUT_DIR, "output_before_pdfs.json")
EXCEL_FILE = os.path.join(OUTPUT_DIR, "rag_chunks_optimized.xlsx")

# --- Chunking ---
CHUNK_SIZE_OPTIMIZED = 800
CHUNK_OVERLAP_OPTIMIZED = 150
CHUNK_SIZE_GRANULAR = 400
CHUNK_OVERLAP_GRANULAR = 80

# --- VLM ---
VLM_PROVIDER_UBICLOUD = "ubicloud"
VLM_PROVIDER_OPENAI = "openai"
VLM_MODEL_UBICLOUD = "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"
VLM_MODEL_OPENAI = "gpt-4o"
