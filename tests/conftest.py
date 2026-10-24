"""Import the pack as a package despite the hyphen in its folder name.

Run from the ComfyUI root with the ComfyUI environment's Python so the
comfy, folder_paths and node_helpers imports in nodes.py resolve:

    python -m pytest custom_nodes/ComfyUI-Kandinsky6-Motion-Context/tests -v
"""

import os
import sys
import types

_PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if "k6mc" not in sys.modules:
    pkg = types.ModuleType("k6mc")
    pkg.__path__ = [_PACK_DIR]
    sys.modules["k6mc"] = pkg
