# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.
"""Fetch peg/hole geometry repositories into a shared local cache.

A ``robot_descriptions``-style loader (mirrors ``assembly.utils.descriptions._cache``): the
mesh/URDF assets live in a standalone git repo (``kamiradi/assembly_description``) rather than
being vendored here, so any package that needs them clones the same cache once and shares it.
``peg_insertion_twist`` uses this to resolve ``MESH_ROOT`` at import instead of hardcoding an
absolute path.

The clone lands in ``~/.cache/descriptions/<name>`` (override the root with the
``ROBOT_DESCRIPTIONS_CACHE`` env var -- the same variable and default the ``assembly`` package
uses, so the two share one cache). ``commit`` is ``None`` by default: the clone tracks the
repo's latest commit and is pulled on each fetch; pin a specific commit only via the
``ROBOT_DESCRIPTION_COMMIT`` env var when reproducibility demands it.

Fetching is explicit and separate from reading, so that a multirun sweep fanning out over N
worker processes does not clone/pull N times (each worker imports the task module afresh):

* ``fetch_descriptions`` / ``python -m judo.descriptions`` -- the only network call.
  Run it once, up front (the sweep scripts do).
* ``get_description_path`` -- pure cache lookup, what task modules call at import. Raises if
  the cache is cold rather than quietly fetching.
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
    is only checked out when given; ``None`` means "track the latest commit", so an existing
    clone is pulled (best-effort -- a failed pull falls back to the cached contents).
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
    elif commit is None:
        # Unpinned: an existing clone would otherwise stay at whatever it was first fetched
        # at, so pull the latest commit. Best-effort -- offline workers keep using the cache.
        try:
            clone.remotes.origin.pull()
        except (GitCommandError, AttributeError, ValueError) as exn:
            print(f"[descriptions] could not pull {target_dir}, using the cached clone: {exn}")

    if commit is not None and commit != clone.head.object.hexsha:
        try:
            clone.git.checkout(commit)
        except GitCommandError:
            print(f"[descriptions] commit {commit} not found, fetching origin and retrying...")
            clone.git.fetch("origin")
            clone.git.checkout(commit)

    return clone


def _repository(description_name: str) -> Repository:
    """Registry lookup for ``description_name``."""
    try:
        return REPOSITORIES[description_name]
    except KeyError as exn:
        raise ImportError(f"unknown description: {description_name!r}") from exn


def _target_dir(description_name: str, commit: str | None = None) -> str:
    """Where ``description_name`` lives (or would live) in the cache, pinned or not."""
    cache_path = _repository(description_name).cache_path
    if commit is not None:
        # Keep the package-named leaf dir so package:// URIs still resolve under the commit dir.
        cache_path = f"{cache_path}-{commit}/{cache_path}"
    return os.path.join(_cache_root(), cache_path)


def clone_to_cache(description_name: str, commit: str | None = None) -> str:
    """Return a working dir for ``description_name``, cloning/pulling it. HITS THE NETWORK.

    ``commit`` (or the ``ROBOT_DESCRIPTION_COMMIT`` env var, if the caller forwards it) pins a
    specific commit; otherwise the repository's own ``commit`` (``None`` = latest) is used.

    Call this once up-front (see ``fetch_descriptions``), never from module-level task code:
    importing a task must not fetch, or every sweep worker re-fetches the same repo.
    """
    repository = _repository(description_name)
    clone = clone_to_directory(
        repository.url,
        _target_dir(description_name, commit),
        commit=repository.commit if commit is None else commit,
    )
    return str(clone.working_dir)


def get_description_path(description_name: str) -> str:
    """Cached working dir for ``description_name``, WITHOUT any network access.

    The assets must already be in the cache -- warm it once per machine with
    ``fetch_descriptions`` (``python -m judo.descriptions``). This is what task modules
    call at import, so importing a task stays offline and instant.

    Honours ``ROBOT_DESCRIPTION_COMMIT`` for an optional pin, matching the ``assembly`` package.

    Raises:
        FileNotFoundError: if the description has not been fetched into the cache yet.
    """
    commit = os.environ.get("ROBOT_DESCRIPTION_COMMIT")
    target_dir = _target_dir(description_name, commit)
    if not os.path.isdir(os.path.join(target_dir, ".git")):
        raise FileNotFoundError(
            f"description {description_name!r} is not in the cache ({target_dir}).\n"
            f"Fetch it once with:  python -m judo.descriptions {description_name}"
        )
    return target_dir


def fetch_descriptions(*description_names: str) -> dict[str, str]:
    """Clone/pull the named descriptions (all of them by default) into the cache.

    The one entry point that touches the network. Run it once before a sweep fans out; the
    workers then resolve paths from the warm cache via ``get_description_path``.
    """
    names = description_names or tuple(REPOSITORIES)
    commit = os.environ.get("ROBOT_DESCRIPTION_COMMIT")
    return {name: clone_to_cache(name, commit=commit) for name in names}


def clear_cache() -> None:
    """Delete the entire descriptions cache root."""
    shutil.rmtree(_cache_root(), ignore_errors=True)


def main() -> None:
    """CLI warm-up: ``python -m judo.descriptions [name ...]`` (default: all)."""
    import argparse  # noqa: PLC0415  -- only needed when run as a script

    parser = argparse.ArgumentParser(description="Fetch description assets into the local cache.")
    parser.add_argument(
        "names",
        nargs="*",
        metavar="NAME",
        help=f"descriptions to fetch (default: all of {', '.join(REPOSITORIES)})",
    )
    args = parser.parse_args()
    for name, path in fetch_descriptions(*args.names).items():
        print(f"[descriptions] {name} -> {path}")


if __name__ == "__main__":
    main()
