"""
Purpose: Transform a bunch of patches that cause bugs into a SWE-bench style dataset.

Usage: python -m swesmith.harness.valid logs/bug_gen/*_patches.json --workers #
"""

import argparse
import json
import os
import shutil
import threading

from collections import defaultdict
from pathlib import Path
from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    FAIL_TO_PASS,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
)
from swebench.harness.docker_build import close_logger
from tqdm.auto import tqdm
from swesmith.constants import (
    KEY_PATCH,
    KEY_TIMED_OUT,
    LOG_TEST_OUTPUT_PRE_GOLD,
    REF_SUFFIX,
    LOG_DIR_RUN_VALIDATION,
)
from swesmith.harness.grading import get_valid_report
from swesmith.harness.utils import run_patch_in_container, run_threadpool
from swesmith.profiles import registry


def print_report(log_dir: Path) -> None:
    if not log_dir.exists():
        print(f"Log directory does not exist: {log_dir}")
        return

    time_outs, f2p_none, f2p_some, other = 0, 0, 0, 0
    folders = [f for f in os.listdir(log_dir) if (log_dir / f).is_dir()]

    for folder in folders:
        report_file = log_dir / folder / LOG_REPORT
        if report_file.is_file():
            try:
                with open(report_file, "r") as f:
                    report = json.load(f)
                if report.get(KEY_TIMED_OUT):
                    time_outs += 1
                elif len(report.get(FAIL_TO_PASS, [])) > 0:
                    f2p_some += 1
                elif len(report.get(FAIL_TO_PASS, [])) == 0:
                    f2p_none += 1
                else:
                    other += 1
            except Exception:
                other += 1

    print(f"Total instances in {log_dir.name}: {len(folders)}")
    print(f"- Timed out: {time_outs}")
    print(f"- Fail to pass: 0 ({f2p_none}); 1+ ({f2p_some})")
    print(f"- Other: {other}")


def run_validation(instance: dict) -> dict:
    instance_id = instance[KEY_INSTANCE_ID]
    rp = registry.get_from_inst(instance)
    valid_folder = LOG_DIR_RUN_VALIDATION / instance["repo"]

    # Use rp.repo_name for consistent naming between main() and run_validation()
    ref_inst_id = f"{rp.repo_name}{REF_SUFFIX}"
    val_postgold_path = valid_folder / ref_inst_id / LOG_TEST_OUTPUT
    report_path = valid_folder / instance_id / LOG_REPORT

    if rp.min_pregold:
        custom_ref_id = f"{instance[KEY_INSTANCE_ID]}{REF_SUFFIX}"
        logger, timed_out = run_patch_in_container(
            {**instance, KEY_INSTANCE_ID: custom_ref_id},
            instance["repo"],
            LOG_DIR_RUN_VALIDATION,
            rp.timeout,
        )
        close_logger(logger)
        if timed_out:
            logger.info(f"Timed out (pre-gold) for {instance_id}.")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as f:
                json.dump({KEY_TIMED_OUT: True, "timeout": rp.timeout}, f, indent=4)
            shutil.rmtree(valid_folder / custom_ref_id, ignore_errors=True)
            return {"status": "timeout"}

        val_postgold_path = valid_folder / instance_id / LOG_TEST_OUTPUT_PRE_GOLD
        val_postgold_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            valid_folder / custom_ref_id / LOG_TEST_OUTPUT,
            val_postgold_path,
        )
        shutil.rmtree(valid_folder / custom_ref_id, ignore_errors=True)

    logger, timed_out = run_patch_in_container(
        instance,
        instance["repo"],
        LOG_DIR_RUN_VALIDATION,
        rp.timeout,
        patch=instance[KEY_PATCH],
    )

    if timed_out:
        logger.info(f"Timed out for {instance_id}.")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump({KEY_TIMED_OUT: True, "timeout": rp.timeout}, f, indent=4)
        close_logger(logger)
        return {"status": "timeout"}

    val_pregold_path = valid_folder / instance_id / LOG_TEST_OUTPUT
    if not val_pregold_path.exists():
        logger.info(f"Pre-gold for {instance_id} failed to run. Exiting early.")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump({KEY_TIMED_OUT: True, "missing_pregold_output": True}, f, indent=4)
        close_logger(logger)
        return {"status": "fail"}

    logger.info(f"Grading answer for {instance_id}...")
    report = get_valid_report(
        val_pregold_path=val_pregold_path,
        val_postgold_path=val_postgold_path,
        instance=instance,
    )
    logger.info(f"Report: {json.dumps(report)}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    close_logger(logger)
    if len(report.get(FAIL_TO_PASS, [])) == 0:
        return {"status": "0_f2p"}
    else:
        return {"status": "1+_f2p"}


def main(
    bug_patches: str,
    workers: int,
    redo_existing: bool = False,
) -> None:
    print(f"Running validation for {bug_patches}...")
    with open(bug_patches, "r") as f:
        raw_patches = json.load(f)

    if not raw_patches:
        print("No patches found in file.")
        return

    bug_patches_list = [
        {
            **x,
            KEY_PATCH: x.get(KEY_PATCH, x.get(KEY_PREDICTION)),
        }
        for x in raw_patches
    ]
    print(f"Found {len(bug_patches_list)} candidate patches.")

    all_repos = set(bp["repo"] for bp in bug_patches_list)
    completed = []

    for repo in all_repos:
        log_dir_parent = LOG_DIR_RUN_VALIDATION / repo
        log_dir_parent.mkdir(parents=True, exist_ok=True)
        if not redo_existing:
            for folder in os.listdir(log_dir_parent):
                log_report_path = log_dir_parent / folder / LOG_REPORT
                if log_report_path.exists():
                    completed.append(folder)

    if len(completed) > 0:
        print(f"Skipping {len(completed)} instances... (--redo_existing to not skip)")
        bug_patches_list = [x for x in bug_patches_list if x[KEY_INSTANCE_ID] not in completed]

    repo_to_bug_patches = defaultdict(list)
    for bp in bug_patches_list:
        repo_to_bug_patches[bp["repo"]].append(bp)

    print("Will run validation for these images:")
    for repo, patches in repo_to_bug_patches.items():
        print(f"- {repo}: {len(patches)} patches")

    payloads = []
    for repo, repo_bug_patches in repo_to_bug_patches.items():
        rp = registry.get(repo)
        ref_inst = f"{rp.repo_name}{REF_SUFFIX}"
        ref_dir = LOG_DIR_RUN_VALIDATION / repo / ref_inst

        if not rp.min_pregold and not os.path.exists(ref_dir):
            print(f"Running pre-gold for {repo}...")
            logger, timed_out = run_patch_in_container(
                {KEY_INSTANCE_ID: ref_inst},
                repo,
                LOG_DIR_RUN_VALIDATION,
                rp.timeout_ref,
            )
            close_logger(logger)
            if timed_out:
                print(f"Timed out for {repo}, skipping this repo. (Increase timeout_ref?)")
                shutil.rmtree(ref_dir, ignore_errors=True)
                continue

        for bug_patch in repo_bug_patches:
            payloads.append((bug_patch,))

    if len(payloads) == 0:
        print("No patches left to run.")
        for repo in all_repos:
            print_report(LOG_DIR_RUN_VALIDATION / repo)
        return

    stats = {"fail": 0, "timeout": 0, "0_f2p": 0, "1+_f2p": 0}
    pbar = tqdm(total=len(payloads), desc="Validation", postfix=stats)
    lock = threading.Lock()

    def run_validation_with_progress(*args):
        instance = args[0] if args else {}
        result = run_validation(instance)
        with lock:
            stats[result["status"]] += 1
            pbar.set_postfix(stats)
            pbar.update()
        return result

    run_threadpool(run_validation_with_progress, payloads, workers)
    pbar.close()

    print("All instances run.")
    for repo in all_repos:
        print_report(LOG_DIR_RUN_VALIDATION / repo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transform a bunch of patches that cause bugs into a SWE-bench style dataset."
    )
    parser.add_argument(
        "bug_patches",
        type=str,
        help="Json file containing bug patches.",
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=4, help="Number of workers to use."
    )
    parser.add_argument(
        "--redo_existing",
        action="store_true",
        help="Redo completed validation instances.",
    )
    args = parser.parse_args()
    main(**vars(args))