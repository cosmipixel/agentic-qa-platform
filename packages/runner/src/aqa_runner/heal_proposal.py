"""A heal proposal (ADR-0024's #243 and #244 amendments, DATA_MODEL §7): what a
run that healed asks a person to accept, as `replay.json` holds it and as `aqa
run` prints it. Nothing applies it: `aqa heal apply` is #247's."""

import difflib

from aqa_core.compiled import CompiledScript
from aqa_core.heal_patch import json_patch, proposal_id

from aqa_runner.publication import published_text
from aqa_runner.repairs import Repaired


def final_script(
    committed: CompiledScript, repairs: tuple[Repaired, ...]
) -> CompiledScript:
    """The script a run went on with last: its last taken repair's, else the
    committed one."""
    taken = [repaired.script for repaired in repairs if repaired.script is not None]
    return taken[-1] if taken else committed


def proposal_document(
    spec_id: str, base: str, committed: CompiledScript, final: CompiledScript
) -> dict[str, object]:
    """The proposal `replay.json` holds: its content-derived id, the compiled
    file it applies to, the blob SHA that file had, with no PR head locally
    (M5's), and the patch."""
    patch = json_patch(committed, final)
    return {
        "id": proposal_id(spec_id, base, patch),
        "compiled": f".compiled/{spec_id}.json",
        "base": {"blob_sha": base, "pr_head_sha": None},
        "patch": patch,
    }


def proposal_lines(
    spec_id: str,
    proposal: str,
    committed: CompiledScript,
    repairs: tuple[Repaired, ...],
) -> tuple[str, ...]:
    """What `aqa run` prints of a proposal: its id, each rebound target's meaning
    and the element it is bound to, then the diff. The role and name are the
    page's own ARIA, so a decoy can match them, and a reviewer's evidence is the
    locator diff (ADR-0024's #244 amendment)."""
    final = final_script(committed, repairs)
    bound: dict[str, str] = {}
    previous = committed
    for repaired in repairs:
        if (script := repaired.script) is None:
            continue
        said = "" if repaired.basis is None else repaired.basis.said
        for name, target in script.targets.items():
            if target.locators != previous.targets[name].locators:
                bound[name] = said
        previous = script
    return (
        f"heal proposal {proposal}:",
        *(
            f"{name} ({target.semantic}): bound {bound[name]}"
            for name, target in committed.targets.items()
            if final.targets[name].locators != target.locators
        ),
        *diff(spec_id, committed, final),
    )


def diff(spec_id: str, committed: CompiledScript, final: CompiledScript) -> list[str]:
    """The patch as a person reads it: a unified diff of the published file,
    each line its own, which the output boundary scans and escapes."""
    path = f".compiled/{spec_id}.json"
    return list(
        difflib.unified_diff(
            published_text(committed).splitlines(),
            published_text(final).splitlines(),
            f"a/{path}",
            f"b/{path}",
            lineterm="",
        )
    )
