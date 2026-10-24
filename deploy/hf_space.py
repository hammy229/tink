"""Deploy Tink to a free Hugging Face Space (Docker SDK, 2 vCPU / 16 GB RAM, HTTPS included).

One-time setup (stores the token on this machine, never in the repo):
    uvx --from huggingface_hub hf auth login          # a "Write" token from huggingface.co/settings/tokens

Then, from the repo root:
    uv run python deploy/hf_space.py                  # Space name defaults to <your-username>/tink
    uv run python deploy/hf_space.py --name tink-hacktx --dry-run

What it does: copies the committed files (git archive HEAD, so local recordings and secrets never ship),
gives the copy a Space README front matter, creates the Space if needed, sets TINK_PUBLIC=1, copies
XAI_API_KEY / TINK_GROK_MODEL from your environment into Space secrets/variables if set, and uploads.
Recordings on a free Space reset when it restarts; that is fine for a demo.
"""
import argparse
import io
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONT_MATTER = """---
title: Tink
emoji: 🔔
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Says what a sound means, in any language
---

"""


def stage(dest: Path) -> None:
    """Committed files only, plus the Space front matter on top of the README."""
    archive = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT, check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")
    readme = dest / "README.md"
    readme.write_text(FRONT_MATTER + readme.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="tink", help="Space name under your account (default: tink)")
    ap.add_argument("--dry-run", action="store_true", help="stage the files and list them, upload nothing")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)
        stage(dest)
        files = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
        print(f"staged {len(files)} committed files")
        if args.dry_run:
            print("\n".join(files))
            return

        from huggingface_hub import HfApi

        api = HfApi()
        user = api.whoami()["name"]
        repo_id = f"{user}/{args.name}"
        api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
        api.add_space_variable(repo_id, "TINK_PUBLIC", "1")
        if os.environ.get("TINK_GROK_MODEL"):
            api.add_space_variable(repo_id, "TINK_GROK_MODEL", os.environ["TINK_GROK_MODEL"])
        if os.environ.get("XAI_API_KEY"):
            api.add_space_secret(repo_id, "XAI_API_KEY", os.environ["XAI_API_KEY"])
            print("added XAI_API_KEY as a Space secret")
        api.upload_folder(folder_path=str(dest), repo_id=repo_id, repo_type="space",
                          commit_message="Deploy Tink", delete_patterns=["*"])
        print(f"pushed. Build logs and app: https://huggingface.co/spaces/{repo_id}")
        print(f"direct app URL (after the build finishes): https://{user}-{args.name}.hf.space".replace("_", "-").lower())


if __name__ == "__main__":
    main()
