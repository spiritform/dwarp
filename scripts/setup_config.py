"""Ask once where the models live and write config.json. Used by install.bat."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
cfg_path = HERE / "config.json"

if cfg_path.is_file():
    root = json.loads(cfg_path.read_text(encoding="utf-8")).get("models_root", "")
    print(f"  config.json already points at: {root}")
    sys.exit(0)

default = HERE / "models"
print()
print("  Where are your Stable Diffusion models?")
print("  Point at a folder that has checkpoints\\, controlnet\\ and upscale_models\\ inside -")
print("  a ComfyUI 'models' folder works as-is (e.g. C:\\ComfyUI\\models).")
print(f"  Press Enter to use DWARP's own folder: {default}")
answer = input("  models folder: ").strip().strip('"')
root = Path(answer) if answer else default

if root == default:   # our own folder: create the layout so it's obvious where files go
    for sub in ("checkpoints", "controlnet", "upscale_models"):
        (root / sub).mkdir(parents=True, exist_ok=True)
if not (root / "checkpoints").is_dir():
    print(f"  note: {root} has no checkpoints\\ folder - DWARP will find no models there yet.")

cfg_path.write_text(json.dumps({"models_root": root.as_posix()}, indent=2), encoding="utf-8")
print(f"  wrote config.json -> {root.as_posix()}")
