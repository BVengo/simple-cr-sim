from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


ROOT_DIR = Path(__file__).parent  # Path to package root
DATA_DIR = ROOT_DIR / "data"

_out = os.getenv("OUT_DIR")
if _out is not None:
    if _out.startswith("./"):
        # Relative path to project root
        OUT_DIR = ROOT_DIR.parent / _out[2:]
    else:
        # Absolute path
        OUT_DIR = Path(_out)
else:
    raise EnvironmentError("Missing OUT_DIR from .env file!")
