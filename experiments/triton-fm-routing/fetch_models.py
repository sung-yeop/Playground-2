"""Download the two public model revisions used in the routing experiment."""

import json
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = {
    "text": (
        "Qwen/Qwen2.5-1.5B-Instruct",
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        [
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "model.safetensors",
        ],
    ),
    "image": (
        "facebook/dinov2-small",
        "ed25f3a31f01632728cabb09d1542f84ab7b0056",
        ["config.json", "preprocessor_config.json", "model.safetensors"],
    ),
}

if __name__ == "__main__":
    models = {}
    for name, (repo, revision, files) in MODELS.items():
        path = Path(
            snapshot_download(repo_id=repo, revision=revision, allow_patterns=files)
        )
        for filename in files:
            if not (path / filename).is_file():
                raise FileNotFoundError(path / filename)
        models[name] = {"repo": repo, "revision": revision, "path": str(path.resolve())}
    Path("models.json").write_text(json.dumps(models, indent=2))
