# arscontexta/.arkhe/utils.py
import sys
from pathlib import Path
import importlib.util

def load_arkhe_module(module_path: Path, module_name: str):
    """Carrega um módulo Arkhe(N) de forma dinâmica."""
    if not module_path.exists():
        raise FileNotFoundError(f"Module not found at {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None:
        raise ImportError(f"Could not load spec for {module_name} at {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def get_arkhe_root() -> Path:
    """Retorna a raiz do projeto arscontexta."""
    # Assume que este arquivo está em arscontexta/.arkhe/utils.py
    return Path(__file__).parent.parent.parent
