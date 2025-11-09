import logging
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv


def safe_load_env(
    *,
    paths: Optional[Iterable[str | Path]] = None,
    override: bool = False,
) -> None:
    """Load environment variables without failing on permission errors."""
    candidate_paths = list(paths or [])
    if not candidate_paths:
        candidate_paths = [Path(".env")]

    for candidate in candidate_paths:
        try:
            load_dotenv(dotenv_path=candidate, override=override)
        except PermissionError:
            logging.warning("Skipping .env file due to permission error: %s", candidate)
        except OSError as exc:
            logging.debug("Unable to load %s: %s", candidate, exc)

