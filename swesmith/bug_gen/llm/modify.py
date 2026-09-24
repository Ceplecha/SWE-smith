"""
Purpose: Given a repository profile, generate bug patches for functions/classes/objects.
Config YAMLs are resolved relative to the target repository under:
    configs/bug_gen/<repo_name>/

Usage:
    # 1. Invoked by bug_gen.py (accepts -c):
    python -m swesmith.bug_gen.llm.modify Ceplecha__pymap-test.9adeb487 -c configs/bug_gen/class_basic.yml

    # 2. Target specific strategy by name in the repo folder:
    python -m swesmith.bug_gen.llm.modify Ceplecha__pymap-test.9adeb487 -s class_basic

    # 3. Automatically run all strategies in configs/bug_gen/pymap-test/ (or pymap/):
    python -m swesmith.bug_gen.llm.modify --repo Ceplecha/pymap-test --commit 9adeb487
"""

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any
import jinja2
import litellm
import yaml

from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from litellm import completion
from litellm.cost_calculator import completion_cost
from swesmith.bug_gen.llm.utils import PROMPT_KEYS, extract_code_block
from swesmith.bug_gen.utils import (
    apply_code_change,
    get_bug_directory,
    get_patch,
)
from swesmith.constants import (
    LOG_DIR_BUG_GEN,
    PREFIX_BUG,
    PREFIX_METADATA,
    BugRewrite,
    CodeEntity,
)
from swesmith.profiles import registry
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

load_dotenv(dotenv_path=os.getenv("SWEFT_DOTENV_PATH"))

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
litellm.suppress_debug_info = True


def resolve_target_and_repo_name(
    target: str | None, repo: str | None, commit: str | None
) -> tuple[str, str]:
    """
    Standardizes inputs into:
      target_id: e.g. 'Ceplecha__pymap-test.9adeb487'
      repo_name: e.g. 'pymap-test'
    """
    if repo and commit:
        cleaned_repo = repo.strip()
        repo_name = cleaned_repo.split("/")[-1].split("__")[-1]
        normalized_repo = cleaned_repo.replace("/", "__")
        short_commit = commit.strip()[:8]
        return f"{normalized_repo}.{short_commit}", repo_name

    if target:
        cleaned = target.strip()
        repo_part = cleaned.rsplit(".", 1)[0] if "." in cleaned else cleaned
        repo_name = repo_part.split("/")[-1].split("__")[-1]

        if "." in cleaned:
            repo_id, commit_id = cleaned.rsplit(".", 1)
            normalized_target = f"{repo_id.replace('/', '__')}.{commit_id[:8]}"
        else:
            normalized_target = repo_part.replace("/", "__")

        return normalized_target, repo_name

    raise ValueError("Specify either target ID or both --repo and --commit.")


def discover_configs_for_repo(
    repo_name: str,
    specific_strategy: str | None = None,
    direct_config_path: str | None = None,
) -> list[Path]:
    """
    Locates strategy YAML files prioritizing the repo-specific subfolder:
        configs/bug_gen/<repo_name>/<strategy>.yml
        configs/bug_gen/<repo_name_without_suffix>/<strategy>.yml
    """
    script_dir = Path(__file__).resolve().parent
    search_roots = [
        Path.cwd(),
        script_dir.parent.parent.parent,  # SWE-smith root
    ]

    candidate_dir_names = [repo_name]
    for suffix in ("-test", "_test", "-main", "_main"):
        if repo_name.endswith(suffix):
            candidate_dir_names.append(repo_name[: -len(suffix)])

    # Case 1: Caller passed an explicit path via -c / --config_file
    if direct_config_path:
        direct_p = Path(direct_config_path)
        filename = direct_p.name

        # 1. Check if the repo subfolder has a file with this name: configs/bug_gen/<repo>/filename
        for root in search_roots:
            for d_name in candidate_dir_names:
                repo_specific = root / "configs" / "bug_gen" / d_name / filename
                if repo_specific.is_file():
                    return [repo_specific.resolve()]

        # 2. Check if the exact path given exists on disk
        if direct_p.is_file():
            return [direct_p.resolve()]

        # 3. Check fallback in general configs/bug_gen/
        for root in search_roots:
            general_cand = root / "configs" / "bug_gen" / filename
            if general_cand.is_file():
                return [general_cand.resolve()]

        raise FileNotFoundError(
            f"Could not locate configuration file '{direct_config_path}' in repo subdirectories "
            f"({candidate_dir_names}) or root configs."
        )

    # Locate the target repo's config directory
    config_dir = None
    for root in search_roots:
        for d_name in candidate_dir_names:
            possible_dir = root / "configs" / "bug_gen" / d_name
            if possible_dir.is_dir():
                config_dir = possible_dir
                break
        if config_dir:
            break

    # Fallback to general configs/bug_gen if no repo-specific directory exists
    if not config_dir:
        for root in search_roots:
            general_dir = root / "configs" / "bug_gen"
            if general_dir.is_dir():
                config_dir = general_dir
                break

    if not config_dir:
        raise FileNotFoundError("Could not locate 'configs/bug_gen/' directory.")

    # Case 2: Specific strategy specified by name (-s / --strategy)
    if specific_strategy:
        strategy_file = Path(specific_strategy)
        matched = []
        for name in (strategy_file.name, strategy_file.with_suffix(".yml").name):
            cand = config_dir / name
            if cand.is_file():
                matched.append(cand.resolve())
                break
        if not matched:
            raise FileNotFoundError(f"Strategy '{specific_strategy}' not found in {config_dir}")
        return matched

    # Case 3: Automatic discovery of all YAML files in the repo config folder
    configs = sorted(list(config_dir.glob("*.yml")) + list(config_dir.glob("*.yaml")))
    if not configs:
        raise FileNotFoundError(f"No .yml or .yaml files found in {config_dir}")

    return [c.resolve() for c in configs]


def gen_bug_from_code_lm(
    candidate: CodeEntity, configs: dict, n_bugs: int, model: str
) -> list[BugRewrite]:
    """Given candidate code entity, requests LLM rewrites to inject bugs."""

    def format_prompt(prompt: str | None, config: dict, candidate: CodeEntity) -> str:
        if not prompt:
            return ""
        env = jinja2.Environment()

        def jinja_shuffle(seq):
            res = list(seq)
            random.shuffle(res)
            return res

        env.filters["shuffle"] = jinja_shuffle
        template = env.from_string(prompt)

        candidate_dict = {
            field.name: getattr(candidate, field.name)
            for field in dataclasses.fields(candidate)
        }
        return template.render(**candidate_dict, **config.get("parameters", {}))

    def get_role(key: str) -> str:
        return "system" if key == "system" else "user"

    bugs = []
    messages = [
        {"content": format_prompt(configs[k], configs, candidate), "role": get_role(k)}
        for k in PROMPT_KEYS
    ]
    messages = [x for x in messages if x["content"]]

    response: Any = completion(model=model, messages=messages, n=n_bugs, temperature=1)
    for choice in response.choices:
        message = choice.message
        explanation = (
            message.content.split("Explanation:")[-1].strip()
            if "Explanation" in message.content
            else message.content.split("```")[-1].strip()
        )
        try:
            cost = completion_cost(completion_response=response) / n_bugs
        except Exception:
            cost = 0.0

        bugs.append(
            BugRewrite(
                rewrite=extract_code_block(message.content),
                explanation=explanation,
                cost=cost,
                output=message.content,
                strategy="llm",
            )
        )
    return bugs


def run_strategy(
    strategy_path: Path,
    candidates: list[CodeEntity],
    target_id: str,
    repo_dir: Path,
    model: str,
    n_bugs: int,
    n_workers: int,
    log_dir: Path,
    max_bugs: int,
):
    configs = yaml.safe_load(strategy_path.read_text(encoding="utf-8"))
    assert all(key in configs for key in PROMPT_KEYS + ["name"]), f"Missing keys in {strategy_path}"

    print(f"\n[STRATEGY] Running strategy: {strategy_path.name} from {strategy_path.parent.name}/")

    if max_bugs > 0:
        max_candidates = max(1, max_bugs // n_bugs)
        selected_candidates = candidates[:max_candidates]
    else:
        selected_candidates = candidates

    def _process_candidate(candidate: CodeEntity):
        bugs = gen_bug_from_code_lm(candidate, configs, n_bugs, model)
        cost = sum(x.cost for x in bugs)
        n_bugs_generated = 0
        n_generation_failed = 0

        for bug in bugs:
            bug_dir = get_bug_directory(log_dir, candidate)
            bug_dir.mkdir(parents=True, exist_ok=True)
            uuid_str = f"{configs['name']}__{bug.get_hash()}"
            metadata_path = f"{PREFIX_METADATA}__{uuid_str}.json"
            bug_path = f"{PREFIX_BUG}__{uuid_str}.diff"

            try:
                with open(bug_dir / metadata_path, "w", encoding="utf-8") as f:
                    json.dump(bug.to_dict(), f, indent=2)

                apply_code_change(candidate, bug)

                patch = get_patch(target_id, reset_changes=True)
                if not patch and repo_dir.exists():
                    patch = get_patch(str(repo_dir), reset_changes=True)

                if not patch:
                    raise ValueError("Generated patch is empty.")

                with open(bug_dir / bug_path, "w", encoding="utf-8") as f:
                    f.write(patch)
            except Exception as e:
                print(f"Error applying bug to {candidate.name} in {candidate.file_path}: {e}")
                (bug_dir / metadata_path).unlink(missing_ok=True)
                n_generation_failed += 1
                continue
            else:
                n_bugs_generated += 1

        return {
            "cost": cost,
            "n_bugs_generated": n_bugs_generated,
            "n_generation_failed": n_generation_failed,
        }

    stats = {"cost": 0.0, "n_bugs_generated": 0, "n_generation_failed": 0}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(_process_candidate, candidate) for candidate in selected_candidates
        ]

        with logging_redirect_tqdm():
            with tqdm(total=len(selected_candidates), desc=f"Candidates ({strategy_path.stem})") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    for k, v in res.items():
                        stats[k] += v
                    pbar.set_postfix(stats, refresh=True)
                    pbar.update(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Target identifier (e.g. 'Ceplecha__pymap-test.9adeb487').",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Target repo ('owner/repo' or 'owner__repo').",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default=None,
        help="Target commit hash.",
    )
    parser.add_argument(
        "-c",
        "--config_file",
        "--config",
        dest="config_file",
        type=str,
        default=None,
        help="Direct path or filename of strategy YAML config.",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        default=None,
        help="Optional single strategy name (e.g. 'class_basic').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="LiteLLM model string.",
    )
    parser.add_argument(
        "-n",
        "--n_bugs",
        type=int,
        default=1,
        help="Bugs per entity.",
    )
    parser.add_argument(
        "-m",
        "--max_bugs",
        type=int,
        default=-1,
        help="Max bugs total per strategy.",
    )
    parser.add_argument(
        "-w",
        "--n_workers",
        type=int,
        default=1,
        help="Worker threads.",
    )

    args = parser.parse_args()
    target_id, repo_name = resolve_target_and_repo_name(args.target, args.repo, args.commit)

    # 1. Discover configs: prioritizes configs/bug_gen/<repo_name>/<strategy>.yml
    strategy_files = discover_configs_for_repo(
        repo_name=repo_name,
        specific_strategy=args.strategy,
        direct_config_path=args.config_file,
    )
    print(f"Target: {target_id}")
    print(f"Discovered strategy config(s): {[str(p) for p in strategy_files]}")

    # 2. Retrieve profile from registry
    try:
        rp = registry.get(target_id)
    except KeyError:
        prefix = target_id.split(".")[0]
        try:
            rp = registry.get(prefix)
        except KeyError:
            print(f"[ERROR] No profile registered for '{target_id}' or '{prefix}'.", file=sys.stderr)
            sys.exit(1)

    # 3. Clone repository
    print(f"Cloning {target_id}...")
    try:
        clone_res = rp.clone(dest=target_id)
    except TypeError:
        clone_res = rp.clone()

    repo_dir = (
        Path(clone_res[0])
        if isinstance(clone_res, tuple)
        else Path(clone_res or target_id)
    )

    # 4. Extract candidate entities
    print("Extracting candidate code entities...")
    candidates = rp.extract_entities()

    ignored_patterns = ("test", "tests", "benchmark", "benchmarks", "docs", "example", "examples")
    candidates = [
        c for c in candidates
        if not any(seg.lower() in ignored_patterns for seg in c.file_path.split("/"))
        and not c.file_path.endswith(("setup.py", "conftest.py"))
    ]

    print(f"{len(candidates)} candidate code entities found.")
    if not candidates:
        if repo_dir.is_dir():
            shutil.rmtree(repo_dir)
        return

    # 5. Set up logs directory and execute each strategy
    log_dir = LOG_DIR_BUG_GEN / target_id
    log_dir.mkdir(parents=True, exist_ok=True)

    for strat_path in strategy_files:
        run_strategy(
            strategy_path=strat_path,
            candidates=candidates,
            target_id=target_id,
            repo_dir=repo_dir,
            model=args.model,
            n_bugs=args.n_bugs,
            n_workers=args.n_workers,
            log_dir=log_dir,
            max_bugs=args.max_bugs,
        )

    # 6. Cleanup clone
    if repo_dir.is_dir():
        shutil.rmtree(repo_dir)


if __name__ == "__main__":
    main()