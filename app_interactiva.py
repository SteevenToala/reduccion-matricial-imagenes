"""Redirecciona a la nueva app web en 3 columnas (Flask + Vue 3)."""
import subprocess
import sys
import os

if __name__ == "__main__":
    ruta_app = os.path.join(os.path.dirname(__file__), "app.py")
    subprocess.run([sys.executable, ruta_app])
