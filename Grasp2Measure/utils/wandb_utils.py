from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def init_wandb(
    enabled: bool,
    config: dict[str, Any],
    run_name: str,
    tags: list[str] | None = None,
):
    if not enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed. Install it in this environment or run without --wandb."
        ) from exc

    wandb_config = config.get("wandb", {})
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True)

    return wandb.init(
        entity=wandb_config.get("entity"),
        project=wandb_config.get("project", "VET6_tactile_model"),
        name=run_name,
        group=wandb_config.get("group"),
        job_type=wandb_config.get("job_type", "train"),
        mode=wandb_config.get("mode", "online"),
        tags=tags or wandb_config.get("tags"),
        config=config,
    )


def log_wandb_artifact(run: Any, path: Path, name: str, artifact_type: str) -> None:
    if run is None or not path.exists():
        return
    import wandb

    artifact = wandb.Artifact(name=name, type=artifact_type)
    artifact.add_file(str(path))
    run.log_artifact(artifact)
