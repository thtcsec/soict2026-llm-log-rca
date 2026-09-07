"""Regenerate only fig1_architecture.png without touching measured fig2-4."""
import runpy
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location(
    "gen", Path(__file__).with_name("generate_all_soict_figures.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.generate_fig1_architecture()
print("[+] fig1 only")
