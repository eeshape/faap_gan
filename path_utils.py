import os
import sys
from pathlib import Path


def _default_detr_repo() -> Path:
    env_repo = os.environ.get("DETR_REPO")
    if env_repo:
        return Path(env_repo).expanduser()
    here = Path(__file__).resolve()
    # search upwards for a sibling "detr" directory
    for parent in here.parents:
        candidate = parent / "detr"
        if candidate.exists():
            return candidate
    # fallback: assume Desktop/detr relative to package
    return here.parents[1] / "detr"


DETR_REPO = _default_detr_repo()


def ensure_detr_repo_on_path(detr_repo: Path = DETR_REPO) -> Path:
    """
    Make sure the DETR repository is importable (datasets, models, util, etc).
    """
    if str(detr_repo) not in sys.path:
        sys.path.insert(0, str(detr_repo))
    return detr_repo


def default_detr_checkpoint(detr_repo: Path = DETR_REPO) -> Path:
    """
    Default pretrained DETR checkpoint location.
    """
    return Path("/home/dohyeong/Desktop/detr/detr-r50-e632da11.pth")
