from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / 'data'

PROMPT_PATH = PROJECT_ROOT / 'prompts' / 'rag_system_prompt.txt'

if __name__ == '__main__':
    print(PROJECT_ROOT)
