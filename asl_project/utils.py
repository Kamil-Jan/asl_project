import logging
from pathlib import Path
from typing import List, Optional

from dvc.repo import Repo
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def infer_class_names(train_dir: Path) -> List[str]:
    if not train_dir.exists():
        return []
    return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])


def ensure_data_ready(
    data_cfg: DictConfig,
    dvc_target: Optional[str] = None,
    use_local_download: bool = False,
) -> Path:
    dataset_root = Path(data_cfg.root_dir)
    if dataset_root.exists() and any(dataset_root.iterdir()):
        return dataset_root

    if dvc_target:
        logger.info("Dataset missing locally, pulling with DVC from %s", dvc_target)
        try:
            repo = Repo(".")
            if dvc_target:
                repo.pull(targets=[dvc_target])
            else:
                repo.pull()
        except Exception as e:
            logger.warning("Failed to pull data with DVC: %s", e)
            if use_local_download:
                logger.info("Falling back to direct download...")
                download_data(dataset_root)
            else:
                raise RuntimeError(f"DVC pull failed: {e}") from e
    elif use_local_download:
        logger.info("No DVC target specified, downloading directly...")
        download_data(dataset_root)
    else:
        raise FileNotFoundError(
            f"Dataset not found at {dataset_root}. "
            "Specify a DVC target or enable use_local_download."
        )

    if not dataset_root.exists():
        raise RuntimeError(f"Dataset should be at {dataset_root} after download.")
    return dataset_root


def current_git_commit() -> str:
    try:
        from git import Repo as GitRepo

        repo = GitRepo(search_parent_directories=True)
        return repo.head.object.hexsha
    except Exception:
        return "unknown"


def download_data(output_dir: Path) -> None:
    import urllib.request
    import zipfile

    dataset_url = (
        "https://github.com/loicmarie/sign-language-alphabet-recognizer/"
        "releases/download/v1.0/asl_alphabet_train.zip"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "asl_alphabet_train.zip"

    logger.info("Downloading ASL Alphabet Dataset from %s", dataset_url)
    logger.info("This may take a while...")

    try:
        urllib.request.urlretrieve(dataset_url, zip_path)
        logger.info("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        zip_path.unlink()
        logger.info("Dataset downloaded and extracted to %s", output_dir)
    except Exception as e:
        logger.error("Failed to download dataset: %s", e)
        if zip_path.exists():
            zip_path.unlink()
        raise
