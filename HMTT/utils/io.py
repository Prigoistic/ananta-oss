"""
I/O utilities for HMTT.

Handles saving and loading of vocabularies, corpora, and models.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def save_vocab(vocab: Dict[str, int], path: str):
    """
    Save vocabulary to JSON file.
    
    Args:
        vocab: Vocabulary mapping (token -> ID)
        path: Output file path
    """
    output_file = Path(path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path: str) -> Dict[str, int]:
    """
    Load vocabulary from JSON file.
    
    Args:
        path: Input file path
        
    Returns:
        Vocabulary mapping (token -> ID)
    """
    with open(path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    return vocab


def save_corpus(corpus: List[str], path: str):
    """
    Save corpus to text file (one document per line).
    
    Args:
        corpus: List of documents (token sequences)
        path: Output file path
    """
    output_file = Path(path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in corpus:
            f.write(doc + '\n')


def load_corpus(path: str) -> List[str]:
    """
    Load corpus from text file.
    
    Args:
        path: Input file path
        
    Returns:
        List of documents
    """
    corpus = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                corpus.append(line)
    
    return corpus


def save_json(data: Any, path: str):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        path: Output file path
    """
    output_file = Path(path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    """
    Load data from JSON file.
    
    Args:
        path: Input file path
        
    Returns:
        Loaded data
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def save_jsonl(data: List[Dict], path: str):
    """
    Save data to JSONL file (one JSON object per line).
    
    Args:
        data: List of dictionaries
        path: Output file path
    """
    output_file = Path(path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_jsonl(path: str) -> List[Dict]:
    """
    Load data from JSONL file.
    
    Args:
        path: Input file path
        
    Returns:
        List of dictionaries
    """
    data = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    return data


def read_text_file(path: str) -> str:
    """
    Read text file.
    
    Args:
        path: Input file path
        
    Returns:
        File contents
    """
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def write_text_file(text: str, path: str):
    """
    Write text to file.
    
    Args:
        text: Text to write
        path: Output file path
    """
    output_file = Path(path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)


def list_files(directory: str, pattern: str = "*") -> List[str]:
    """
    List files in directory matching pattern.
    
    Args:
        directory: Directory path
        pattern: Glob pattern (default: all files)
        
    Returns:
        List of file paths
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        return []
    
    return [str(p) for p in dir_path.glob(pattern) if p.is_file()]


def ensure_dir(path: str):
    """
    Ensure directory exists.
    
    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def file_exists(path: str) -> bool:
    """
    Check if file exists.
    
    Args:
        path: File path
        
    Returns:
        True if file exists
    """
    return Path(path).exists()


def get_file_size(path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        path: File path
        
    Returns:
        File size in bytes
    """
    return Path(path).stat().st_size


def get_file_lines(path: str) -> int:
    """
    Count number of lines in file.
    
    Args:
        path: File path
        
    Returns:
        Number of lines
    """
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)
