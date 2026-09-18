"""
Purpose: Given the validation logs, create a SWE-bench-style dataset + set of repositories
that can be run with SWE-agent. Each instance is of the form:

{
    "instance_id":
    "repo":
    "patch":
    "test_patch":
    "problem_statement":
    "PASS_TO_FAIL":
    "PASS_TO_PASS":
    "version":
}

This script will clone the repository, apply the patches and push them to new branches.

IMPORTANT: Make sure you run authenticated git, because else you'll get rate limit issues.

Usage: python -m swesmith.harness.gather logs/run_validation/<run_id>
"""

import argparse
import concurrent.futures
import functools
import json
import os
from pathlib import Path
import shlex
import subprocess

from swebench.harness.constants import (
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    LOG_REPORT,
    PASS_TO_FAIL,
    PASS_TO_PASS,
)
from swesmith.constants import (
    GIT_APPLY_CMDS,
    KEY_IMAGE_NAME,
    KEY_PATCH,
    KEY_TIMED_OUT,
    LOG_DIR_RUN_VALIDATION,
    LOG_DIR_TASKS,
    REF_SUFFIX,
)
from swesmith.profiles import registry
from tqdm.auto import tqdm

FAILURE_TIPS = """
IMPORTANT

1. If this script fails, you might have to remove the repo & reclone it or remove all branches. 
   Else you might get issues during git checkout -o . 
   Because some branches exist locally but not pushed to the remote on GitHub.

2. Make sure you run authenticated git, because else you'll get rate limit issues that are 
   interpreted as non-existent branches. Causing issues similar to 1.
"""

SUBPROCESS_ARGS = {
    "check": True,
    "shell": True,
}


def main(*args, **kwargs):
    """
    Main entry point for the script.
    """
    try:
        _main(*args, **kwargs)
    except Exception:
        print("=" * 80)
        print("=" * 80)
        print(FAILURE_TIPS)
        print("=" * 80)
        print("=" * 80)
        raise


def skip_print(reason: str, pbar: tqdm, stats: dict, verbose: bool):
    stats["skipped"] += 1
    pbar.set_postfix(stats)
    if verbose:
        print(f"[SKIP] {reason}")
    pbar.update()
    return stats


def check_if_branch_exists(
    repo_name: str,
    subfolder: str,
    main_branch: str,
    override_branch: bool,
    verbose: bool,
    subprocess_args: dict,
):
    branch_exists = False
    try:
        result = subprocess.run(
            f"git ls-remote --heads origin {subfolder}",
            cwd=repo_name,
            capture_output=True,
            shell=True,
            text=True,
        )

        if result.returncode == 0 and subfolder in result.stdout:
            branch_exists = True
            if override_branch:
                subprocess.run(
                    f"git push --delete origin {subfolder}",
                    cwd=repo_name,
                    **subprocess_args,
                )
                if verbose:
                    print(f"[{subfolder}] Overriding existing branch")
                branch_exists = False

    except Exception:
        branch_exists = False
        pass
    return branch_exists


def _main(
    validation_logs_path: str | Path,
    *,
    debug_subprocess: bool = False,
    override_branch: bool = False,
    repush_image: bool = False,
    verbose: bool = False,
):
    for var in ["SWE_GITHUB_USER", "SWE_USE_SSH", "SWE_REPO_SUFFIX", "MAX_WORKERS"]:
        if var not in os.environ:
            raise KeyError(f"Mandatory environment variable '{var}' is not set.")

    if not debug_subprocess:
        SUBPROCESS_ARGS["stdout"] = subprocess.DEVNULL
        SUBPROCESS_ARGS["stderr"] = subprocess.DEVNULL

    validation_logs_path = Path(validation_logs_path)
    assert validation_logs_path.resolve().is_relative_to(
        LOG_DIR_RUN_VALIDATION.resolve()
    ), f"Validation logs should be in {LOG_DIR_RUN_VALIDATION}"
    assert validation_logs_path.exists(), (
        f"Validation logs path {validation_logs_path} does not exist"
    )
    assert validation_logs_path.is_dir(), (
        f"Validation logs path {validation_logs_path} is not a directory"
    )

    run_id = validation_logs_path.name
    print(f"{run_id=}")
    task_instances_path = LOG_DIR_TASKS / f"{run_id}.json"
    print(f"Out Path: {task_instances_path}")
    task_instances = []
    created_repos = set()

    completed_ids = []
    subfolders = os.listdir(validation_logs_path)
    if not override_branch and os.path.exists(task_instances_path):
        with open(task_instances_path) as f:
            task_instances = [
                x
                for x in json.load(f)
                if x[KEY_INSTANCE_ID] in subfolders
            ]
        completed_ids = [x[KEY_INSTANCE_ID] for x in task_instances]
        print(f"Found {len(task_instances)} existing task instances")
        subfolders = [x for x in subfolders if x not in completed_ids]

    completed_ids = set(completed_ids)
    subfolders_to_process = [x for x in subfolders if x not in completed_ids]

    print(f"Will process {len(subfolders_to_process)} instances")

    n_workers = int(os.environ["MAX_WORKERS"].strip("\"'"))
    print(f"Using {n_workers} workers")

    import tempfile

    with tempfile.TemporaryDirectory() as cache_root:
        cache_dir = os.path.join(cache_root, "repo")
        print(f"Pre-cloning repository to cache: {cache_dir}...")

        rp_cache = None
        try:
            rp_cache = registry.get(run_id)
        except Exception:
            pass

        if not rp_cache:
            sample_id = next((s for s in subfolders if "." in s), None)
            if sample_id:
                try:
                    rp_cache = registry.get_from_inst({KEY_INSTANCE_ID: sample_id})
                except Exception as e:
                    print(f"Warning: Could not resolve profile from {sample_id}: {e}")

        path_to_cache = None
        if rp_cache:
            try:
                print(f"Cloning {rp_cache.repo_name} to cache...")
                rp_cache.clone(dest=cache_dir)
                path_to_cache = cache_dir
                print("Pre-clone successful.")
            except Exception as e:
                print(f"Pre-clone failed: {e}. Will fall back to per-instance cloning.")
        else:
            print(
                "Could not resolve profile for pre-cloning. Will iterate per instance."
            )

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            func = functools.partial(
                process_instance,
                validation_logs_path=validation_logs_path,
                override_branch=override_branch,
                debug_subprocess=debug_subprocess,
                verbose=verbose,
                cache_dir=path_to_cache,
            )

            results = list(
                tqdm(
                    executor.map(func, sorted(subfolders_to_process)),
                    total=len(subfolders_to_process),
                    desc="Conversion",
                )
            )

    stats = {"new_tasks": 0, "skipped": 0}
    for res_tasks, res_repos, res_stats in results:
        task_instances.extend(res_tasks)
        created_repos.update(res_repos)
        for k, v in res_stats.items():
            stats[k] += v

    if len(created_repos) > 0:
        if repush_image:
            print("Rebuilding + pushing images...")
            for repo in created_repos:
                print(f"[{repo}] Rebuilding + pushing image")
                registry.get(repo).push_image(rebuild_image=True)

    if len(task_instances) > 0:
        task_instances_path.parent.mkdir(parents=True, exist_ok=True)
        with open(task_instances_path, "w") as f:
            json.dump(task_instances, f, indent=4)
        print(f"Wrote {len(task_instances)} instances to {task_instances_path}")

    print(f"- {stats['skipped']} skipped")
    print(f"- {stats['new_tasks']} new instances")


def process_instance(
    subfolder: str,
    validation_logs_path: Path,
    override_branch: bool,
    debug_subprocess: bool,
    verbose: bool,
    cache_dir: str | None = None,
) -> tuple[list[dict], set[str], dict]:
    stats = {"new_tasks": 0, "skipped": 0}
    task_instances = []
    created_repos = set()

    import multiprocessing

    pid = multiprocessing.current_process().pid

    os.environ["GIT_SSH_COMMAND"] = "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"

    subprocess_args = SUBPROCESS_ARGS.copy()
    if not debug_subprocess:
        subprocess_args["stdout"] = subprocess.DEVNULL
        subprocess_args["stderr"] = subprocess.DEVNULL

    if subfolder.endswith(REF_SUFFIX):
        return [], set(), {"new_tasks": 0, "skipped": 1}

    path_results = os.path.join(validation_logs_path, subfolder, LOG_REPORT)
    path_patch = os.path.join(validation_logs_path, subfolder, "patch.diff")

    if not os.path.exists(path_results):
        if verbose:
            print(f"[SKIP] {subfolder}: No results")
        return [], set(), {"new_tasks": 0, "skipped": 1}

    if not os.path.exists(path_patch):
        if verbose:
            print(f"[SKIP] {subfolder}: No patch.diff")
        return [], set(), {"new_tasks": 0, "skipped": 1}

    with open(path_results) as f:
        results = json.load(f)

    f2p_key = FAIL_TO_PASS if FAIL_TO_PASS in results else PASS_TO_FAIL
    p2p_key = PASS_TO_PASS

    if f2p_key not in results or p2p_key not in results:
        if verbose:
            print(f"[SKIP] {subfolder}: No validatable bugs")
        return [], set(), {"new_tasks": 0, "skipped": 1}

    n_f2p = len(results[f2p_key])
    n_p2p = len(results[p2p_key])

    pr_exception = ".pr_" in subfolder and n_p2p == 0 and n_f2p > 0
    if not pr_exception and (KEY_TIMED_OUT in results or n_f2p == 0 or n_p2p == 0):
        if verbose:
            print(f"[SKIP] {subfolder}: No validatable bugs: {n_f2p=}, {n_p2p=}")
        return [], set(), {"new_tasks": 0, "skipped": 1}

    with open(path_patch) as f:
        patch_content = f.read()
    task_instance = {
        KEY_INSTANCE_ID: subfolder,
        KEY_PATCH: patch_content,
        FAIL_TO_PASS: results[f2p_key],
        PASS_TO_PASS: results[p2p_key],
    }
    rp = registry.get_from_inst(task_instance)
    task_instance[KEY_IMAGE_NAME] = rp.image_name
    task_instance["repo"] = rp.mirror_name

    if cache_dir:
        repo_path = os.path.join(
            os.path.dirname(cache_dir), f"{rp.repo_name}_worker_{pid}"
        )
    else:
        repo_path = os.path.abspath(f"{rp.repo_name}_worker_{pid}")

    def reset_repo(path):
        subprocess.run("git reset --hard", cwd=path, **subprocess_args)
        subprocess.run("git clean -fdx", cwd=path, **subprocess_args)
        lock_file = os.path.join(path, ".git", "index.lock")
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except OSError:
                pass

    cloned = False
    try:
        github_user = os.environ["SWE_GITHUB_USER"].strip("\"'")
        use_ssh = os.environ["SWE_USE_SSH"].strip("\"'").lower() in ("1", "true", "yes")
        repo_suffix = os.environ["SWE_REPO_SUFFIX"].strip("\"'")

        # If SWE_REPO_NAME was set explicitly by the caller, use it.
        # Otherwise, derive the base repository name and strip unwanted hash suffixes.
        if "SWE_REPO_NAME" in os.environ:
            base_name = os.environ["SWE_REPO_NAME"].strip("\"'")
        else:
            base_name = rp.repo_name.split("__")[-1].split(".")[0]
            if repo_suffix and not base_name.endswith(repo_suffix):
                base_name = f"{base_name}{repo_suffix}"

        custom_repo = f"{github_user}/{base_name}"

        if use_ssh:
            remote_url = f"git@github.com:{custom_repo}.git"
        else:
            remote_url = f"https://github.com/{custom_repo}.git"

        if os.path.exists(repo_path):
            if verbose:
                print(f"[{subfolder}] Reusing worker repo {repo_path}")
            reset_repo(repo_path)

            main_branch = (
                subprocess.run(
                    "git rev-parse --abbrev-ref HEAD",
                    cwd=repo_path,
                    capture_output=True,
                    shell=True,
                    check=True,
                )
                .stdout.decode()
                .strip()
            )
            subprocess.run(
                f"git checkout {main_branch}", cwd=repo_path, **subprocess_args
            )
        else:
            if cache_dir and os.path.exists(cache_dir):
                if verbose:
                    print(f"[{subfolder}] First-time clone from cache {cache_dir}...")

                subprocess.run(
                    f"git clone {cache_dir} {repo_path}",
                    check=True,
                    shell=True,
                    stdout=subprocess.DEVNULL if not debug_subprocess else None,
                    stderr=subprocess.DEVNULL if not debug_subprocess else None,
                )
                cloned = True
                created_repos.add(rp.repo_name)
            else:
                _, cloned = rp.clone(dest=repo_path)
                created_repos.add(rp.repo_name)

            main_branch = (
                subprocess.run(
                    "git rev-parse --abbrev-ref HEAD",
                    cwd=repo_path,
                    capture_output=True,
                    shell=True,
                    check=True,
                )
                .stdout.decode()
                .strip()
            )

        # Force origin to point to the resolved GitHub remote
        subprocess.run(
            f"git remote set-url origin {remote_url}",
            cwd=repo_path,
            check=True,
            shell=True,
            stdout=subprocess.DEVNULL if not debug_subprocess else None,
            stderr=subprocess.DEVNULL if not debug_subprocess else None,
        )

        subprocess.run(f"git checkout {main_branch}", cwd=repo_path, **subprocess_args)

        branch_exists = check_if_branch_exists(
            repo_path, subfolder, main_branch, override_branch, verbose, subprocess_args
        )
        if branch_exists:
            task_instances.append(task_instance)
            if verbose:
                print(f"[SKIP] {subfolder}: Branch `{subfolder}` exists")
            stats["skipped"] += 1
            return task_instances, created_repos, stats

        elif verbose:
            print(f"[{subfolder}] Does not exist yet")

        applied = False
        abs_patch_path = shlex.quote(os.path.abspath(path_patch))
        for git_apply in GIT_APPLY_CMDS:
            output = subprocess.run(
                f"{git_apply} {abs_patch_path}",
                cwd=repo_path,
                capture_output=True,
                shell=True,
            )
            if output.returncode == 0:
                applied = True
                break
            else:
                subprocess.run("git reset --hard", cwd=repo_path, **subprocess_args)

        if not applied:
            print(f"[{subfolder}] Failed to apply patch to {rp.repo_name}")
            reset_repo(repo_path)
            return [], set(), stats

        if verbose:
            print(f"[{subfolder}] Bug patch applied successfully")

        cmds = [
            "git config user.email 'swesmith@swesmith.ai'",
            "git config user.name 'swesmith'",
            "git config commit.gpgsign false",
            f"git checkout -b {subfolder}",
            "git add .",
        ]
        for cmd in cmds:
            if debug_subprocess:
                print(f"[{subfolder}] {cmd}")
            subprocess.run(cmd, cwd=repo_path, **subprocess_args)

        status_output = (
            subprocess.run(
                "git status --porcelain",
                cwd=repo_path,
                capture_output=True,
                shell=True,
                check=True,
            )
            .stdout.decode()
            .strip()
        )

        if not status_output:
            if verbose:
                print(f"[{subfolder}] No changes to commit, skipping")
            stats["skipped"] += 1
            subprocess.run(
                f"git checkout {main_branch}", cwd=repo_path, **subprocess_args
            )
            subprocess.run(
                f"git branch -D {subfolder}", cwd=repo_path, **subprocess_args
            )
            return task_instances, created_repos, stats

        cmds = [
            "git commit --no-gpg-sign -m 'Bug Patch'",
        ]
        for cmd in cmds:
            if debug_subprocess:
                print(f"[{subfolder}] {cmd}")
            subprocess.run(cmd, cwd=repo_path, **subprocess_args)

        f2p_test_files, _ = rp.get_test_files(task_instance)
        if f2p_test_files:
            for test_file in f2p_test_files:
                test_file_path = os.path.join(repo_path, test_file)
                if os.path.exists(test_file_path):
                    os.remove(test_file_path)
                    if verbose:
                        print(f"[{subfolder}] Removed F2P test file: {test_file}")

            cmds = [
                "git add .",
                "git commit --no-gpg-sign -m 'Remove F2P Tests'",
            ]
            for cmd in cmds:
                if debug_subprocess:
                    print(f"[{subfolder}] {cmd}")
                subprocess.run(cmd, cwd=repo_path, **subprocess_args)
            if verbose:
                print(f"[{subfolder}] Commit F2P test file(s) removal")

        push_cmds = [
            f"git push origin {subfolder}",
            f"git checkout {main_branch}",
            "git reset --hard",
            f"git branch -D {subfolder}",
        ]
        for cmd in push_cmds:
            if debug_subprocess:
                print(f"[{subfolder}] {cmd}")
            proc = subprocess.run(
                cmd,
                cwd=repo_path,
                shell=True,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode != 0:
                remote_check = subprocess.run(
                    "git remote get-url origin",
                    cwd=repo_path,
                    shell=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                print(f"\n[FATAL GIT ERROR in {subfolder}]", flush=True)
                print(f"  Command:    {cmd}", flush=True)
                print(f"  Remote URL: {remote_check}", flush=True)
                print(f"  Exit code:  {proc.returncode}", flush=True)
                print(f"  Stderr:     \n{proc.stderr.strip()}", flush=True)
                print(f"  Stdout:     \n{proc.stdout.strip()}\n", flush=True)
                raise subprocess.CalledProcessError(
                    proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr
                )

        if verbose:
            print(f"[{subfolder}] Bug @ branch `{subfolder}`")

        task_instances.append(task_instance)
        if verbose:
            print(f"[{subfolder}] Created task instance")
        stats["new_tasks"] += 1

    finally:
        pass

    return task_instances, created_repos, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert validation logs to SWE-bench style dataset"
    )
    parser.add_argument(
        "validation_logs_path", type=str, help="Path to the validation logs"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose mode",
    )
    parser.add_argument(
        "-o",
        "--override_branch",
        action="store_true",
        help="Override existing branches",
    )
    parser.add_argument(
        "-d",
        "--debug_subprocess",
        action="store_true",
        help="Debug mode (output subprocess output)",
    )
    parser.add_argument(
        "-p",
        "--repush_image",
        action="store_true",
        help="Rebuild and push Docker image for repos (such that latest branches are included)",
    )
    args = parser.parse_args()

    main(**vars(args))