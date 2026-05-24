import os
import shutil
from subprocess import run

venv_dir = os.path.abspath("./stt-venv")
uv_path = shutil.which("uv")
assert uv_path, "uv not found in PATH"

if not os.path.isdir(venv_dir):
    run(
        [uv_path, "sync", "--no-config", "--project", "."],
        env={**os.environ, "UV_PROJECT_ENVIRONMENT": venv_dir},
        check=True,
    )

os.makedirs("private", exist_ok=True)

if not os.path.isfile("private/config.yaml"):
    shutil.copyfile("templates/config.yaml.example", "private/config.yaml")
