"""install.bat: the MiDaS depth model DepthDiff (depth) and the depth ControlNet hints use, fetched into the
Hugging Face cache where the detector looks for it, and checked against its published SHA-256."""
import hashlib
import sys

from huggingface_hub import hf_hub_download

REPO, FILE = "lllyasviel/Annotators", "dpt_hybrid-midas-501f0c75.pt"
SHA256 = "501f0c75b3bca7daec6b3682c5054c09b366765aef6fa3a09d03a5cb4b230853"

path = hf_hub_download(REPO, FILE)
h = hashlib.sha256()
with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
if h.hexdigest() != SHA256:
    sys.exit(f"  {FILE}: checksum mismatch")
print(f"  {FILE}: ok")
