import os
from pathlib import Path
from typing import Union

def save(path: str, file: str, content: Union[str, bytes], mode: str = 'w', encoding: str = 'utf-8'):
    full_path = os.path.join(path, file)
    with open(full_path, mode, encoding=encoding if 'b' not in mode else None) as f:
        f.write(content)

def load(path: str, file: str, mode: str = 'r', encoding: str = 'utf-8') -> Union[str, bytes]:
    full_path = os.path.join(path, file)
    with open(full_path, mode, encoding=encoding if 'b' not in mode else None) as f:
        return f.read()

def validate_file_path(path_str):
    if not path_str:
        return
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path_str}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path_str}")
