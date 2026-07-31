# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.
"""Fetch peg/hole geometry repositories into a shared local cache.

A ``robot_descriptions``-style loader (mirrors ``assembly.utils.descriptions._cache``): the
mesh/URDF assets live in a standalone git repo (``kamiradi/assembly_description``) rather than
being vendored here, so any package that needs them clones the same cache once and shares it.
``peg_insertion_twist`` uses this to resolve ``MESH_ROOT`` at import instead of hardcoding an
absolute path.

The clone lands in ``~/.cache/descriptions/<name>`` (override the root with the
``ROBOT_DESCRIPTIONS_CACHE`` env var -- the same variable and default the ``assembly`` package
uses, so the two share one cache). ``commit`` is ``None`` by default: the first clone resolves
to the repo's latest commit and is reused thereafter; pin a specific commit only via the
``ROBOT_DESCRIPTION_COMMIT`` env var when reproducibility demands it.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from git import GitCommandError, InvalidGitRepositoryError, Repo

DEFAULT_CACHE_DIR = "~/.cache/descriptions"


@dataclass
class Repository:
    """A remote git repository of description assets.

    Attributes:
        url: URL of the remote git repository.
        commit: Commit/tag to check out after cloning, or ``None`` to track the latest commit.
        cache_path: Sub-directory under the cache root to clone into (usually the repo name).
    """

    url: str
    commit: str | None
    cache_path: str


# commit=None -> always resolve to the latest commit (not pinned); see module docstring.
REPOSITORIES: dict[str, Repository] = {
    "assembly_description": Repository(
        url="https://github.com/kamiradi/assembly_description.git",
        commit=None,
        cache_path="assembly_description",
    ),
}


def _cache_root() -> str:
    """Absolute path of the descriptions cache root (``ROBOT_DESCRIPTIONS_CACHE`` overrides)."""
    return os.path.expanduser(os.environ.get("ROBOT_DESCRIPTIONS_CACHE", DEFAULT_CACHE_DIR))


def clone_to_directory(repo_url: str, target_dir: str, commit: str | None = None) -> Repo:
    """Clone ``repo_url`` into ``target_dir`` (reusing an existing valid clone), checkout ``commit``.

    A pre-existing directory that is not a valid git repo is deleted and re-cloned. ``commit``
    is only checked out when given; ``None`` leaves the clone at whatever it was fetched at
    (i.e. the latest commit for a fresh clone).
    """
    clone: Repo | None = None
    if os.path.exists(target_dir):
        try:
            clone = Repo(target_dir)
        except InvalidGitRepositoryError:
            print(f"[descriptions] {target_dir} is not a valid repo, recreating it...")
            shutil.rmtree(target_dir)
            clone = None

    if clone is None:
        print(f"[descriptions] cloning {repo_url} -> {target_dir}")
        os.makedirs(target_dir, exist_ok=True)
        clone = Repo.clone_from(repo_url, target_dir)

    if commit is not None and commit != clone.head.object.hexsha:
        try:
            clone.git.checkout(commit)
        except GitCommandError:
            print(f"[descriptions] commit {commit} not found, fetching origin and retrying...")
            clone.git.fetch("origin")
            clone.git.checkout(commit)

    return clone


def clone_to_cache(description_name: str, commit: str | None = None) -> str:
    """Return a local working directory for ``description_name``, cloning it on first use.

    ``commit`` (or the ``ROBOT_DESCRIPTION_COMMIT`` env var, if the caller forwards it) pins a
    specific commit; otherwise the repository's own ``commit`` (``None`` = latest) is used.
    """
    try:
        repository = REPOSITORIES[description_name]
    except KeyError as exn:
        raise ImportError(f"unknown description: {description_name!r}") from exn

    cache_path = repository.cache_path
    if commit is not None:
        # Keep the package-named leaf dir so package:// URIs still resolve under the commit dir.
        cache_path = f"{cache_path}-{commit}/{cache_path}"
    target_dir = os.path.join(_cache_root(), cache_path)

    clone = clone_to_directory(
        repository.url,
        target_dir,
        commit=repository.commit if commit is None else commit,
    )
    return str(clone.working_dir)


def get_description_path(description_name: str) -> str:
    """Convenience wrapper: cached working dir for ``description_name`` (latest commit).

    Honours ``ROBOT_DESCRIPTION_COMMIT`` for an optional pin, matching the ``assembly`` package.
    """
    return clone_to_cache(description_name, commit=os.environ.get("ROBOT_DESCRIPTION_COMMIT"))


def clear_cache() -> None:
    """Delete the entire descriptions cache root."""
    shutil.rmtree(_cache_root(), ignore_errors=True)
