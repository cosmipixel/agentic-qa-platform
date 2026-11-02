"""The seam between a replay and its repair (ADR-0024's #242 and #244
amendments): what a drift hands over, every check of the script a repair
offers, and the record of each handoff. The executor decides when something
drifted; `Handing` does the rest."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from aqa_core.compiled import (
    Click,
    CompiledScript,
    Fill,
    FillSecret,
    Select,
    Step,
    TextInTarget,
    VisibleUnoccluded,
)
from aqa_core.config import ProjectConfig
from aqa_core.heal_patch import heal_patch_problems, pointer
from aqa_core.project import SpecError, parse_compiled
from aqa_core.spec import Spec
from pydantic_core import PydanticSerializationError

from aqa_runner.admission import Targeted, accepted
from aqa_runner.browser_session import BrowserSession
from aqa_runner.locators import Miss
from aqa_runner.redaction import Redacted
from aqa_runner.run_record import RefusedWriteError, RunRecord, require_clean


@dataclass(frozen=True)
class StepAt:
    """A drifted step: its target didn't resolve, so nothing was dispatched."""

    step: Targeted

    @property
    def key(self) -> str:
        """What its record is named by: `repairs/<seq>.json`."""
        return str(self.step.seq)


@dataclass(frozen=True)
class AssertionAt:
    """A drifted positive check, and its index in the script's assertions."""

    index: int
    check: TextInTarget | VisibleUnoccluded

    @property
    def key(self) -> str:
        """What its record is named by, since a check's id may hold `/`:
        `repairs/assertion-<index>.json`."""
        return f"assertion-{self.index}"


# Where a replay paused for its repair.
type At = StepAt | AssertionAt


@dataclass(frozen=True)
class Drift:
    """Where a replay paused, for its repair: a deep copy of what drifted;
    each locator's miss; a deep copy of the script in use, the committed one
    or an earlier repair's, so an edit to either reaches nothing replay holds;
    and the live session, which the repair observes through and dispatches
    nothing with."""

    at: At
    misses: tuple[Miss, ...]
    script: CompiledScript
    session: BrowserSession


# What a replay hands a drift to: the whole script to go on with. Returning
# the script it was handed gives up, and what drifted stays drifted.
type Repair = Callable[[Drift], Awaitable[CompiledScript]]


@dataclass(frozen=True)
class Repaired:
    """One drift's repair, as `repairs/<at.key>.json` records it: what
    drifted and its misses; why replay refused the script offered, each
    problem scanned, none when it was taken; and the script the run went on
    with, as replay parsed it, None when refused. A repair that offers the
    script it was handed gives up: neither problems nor a script."""

    at: At
    misses: tuple[Miss, ...]
    problems: tuple[Redacted, ...]
    script: CompiledScript | None


@dataclass(frozen=True)
class Handing:
    """One run's handoffs to its repair: the repair; `committed`, replay's
    own copy of the script it was given, which nothing handed to a repair
    holds, since the session holds its browser settings; what the checks and
    the record need; and each handoff so far, in order."""

    repair: Repair
    committed: CompiledScript
    record: RunRecord
    config: ProjectConfig
    spec: Spec
    session: BrowserSession
    repaired: list[Repaired] = field(default_factory=list)

    def handed(self, at: At) -> bool:
        """Whether `at` has had its one repair."""
        return any(repaired.at.key == at.key for repaired in self.repaired)

    async def hand(
        self,
        at: At,
        misses: tuple[Miss, ...],
        in_use: CompiledScript,
        ran: Sequence[Step],
    ) -> Repaired:
        """Give the repair a `Drift` for `at`, with deep copies of it and of
        `in_use`, then check and record what it offers (`_checked`). What the
        repair raises leaves as any failure of the run does. A target a step in
        `ran` acted through goes to no repair and is recorded as refused: no
        offer could change it (`_ran_problems`), so nothing is spent asking."""
        target = at.step.target if isinstance(at, StepAt) else at.check.target
        if target in _acted_through(ran):
            problem = (
                f"/targets/{pointer(target)}: a step that already ran acted "
                "through this target, so it goes to no repair"
            )
            repaired = self._refused(at, misses, [problem])
        else:
            offered = await self.repair(
                Drift(deepcopy(at), misses, in_use.model_copy(deep=True), self.session)
            )
            repaired = await self._checked(at, misses, offered, in_use, ran)
        self.repaired.append(repaired)
        return repaired

    async def _checked(
        self,
        at: At,
        misses: tuple[Miss, ...],
        offered: CompiledScript,
        in_use: CompiledScript,
        ran: Sequence[Step],
    ) -> Repaired:
        """The repair of `at`, whose repair offered `offered`, recorded in
        `repairs/<at.key>.json`: taken, as the loader reads its JSON, once
        nothing refuses it, or given up, with neither problems nor a script,
        when the loader reads it as `in_use`, the script the repair was
        handed a copy of. The first of these to find problems refuses it, and
        each problem is scanned: its JSON, which must hold only values its
        fields' types allow (refused unquoted) and no bound value's spelling;
        the loader's rules, which a `model_copy` skips; the heal-patch
        validator, against `committed`; what ran stays as it ran
        (`_ran_problems`), since the run goes on after it; replay's own
        admission (`accepted`); and the record, which checks the document as
        written and refuses one that spells a bound value, so nothing runs
        that isn't recorded. The refusal is recorded through the same check,
        which raises `RefusedWriteError` when the document still spells a
        value once its problems are scanned: a miss or a scan's marker can.
        The validator searches text checks' patterns in a child process, so it
        runs off the loop, which keeps the proxy serving."""
        record = self.record
        name = f"repairs/{at.key}.json"
        clean = require_clean(record.redactor, name)
        problems: Sequence[str]
        try:
            # A field of a type `model_copy` let through makes the serializer
            # warn, quoting its value: raised here, and its text never kept.
            text = offered.model_dump_json(warnings="error")
            # Before any problem names a part of it: a validator's pointer
            # escapes `/` and `~` (RFC 6901), a spelling no scan finds.
            clean(text)
            patched = parse_compiled(text, self.config, source=Path(name))
            if patched.model_dump(mode="json") == in_use.model_dump(mode="json"):
                gave_up = Repaired(at, misses, (), None)
                record.write(name, _repair_record(gave_up), check=clean)
                return gave_up
            problems = await asyncio.to_thread(
                heal_patch_problems, self.committed, patched
            )
            if not problems:
                problems = _ran_problems(patched, in_use, ran, at)
            if not problems:
                accepted(patched, self.spec)
                taken = Repaired(at, misses, (), patched)
                record.write(name, _repair_record(taken), check=clean)
                return taken
        except PydanticSerializationError:
            problems = [
                f"{name}: a field holds a value its type doesn't allow, so it isn't read"
            ]
        except SpecError as error:
            problems = error.problems
        except RefusedWriteError as error:
            problems = [str(error)]
        return self._refused(at, misses, problems)

    def _refused(
        self, at: At, misses: tuple[Miss, ...], problems: Sequence[str]
    ) -> Repaired:
        """`at`'s repair refused, each of `problems` scanned, and recorded."""
        name = f"repairs/{at.key}.json"
        redactor = self.record.redactor
        scanned = tuple(redactor.redact(problem) for problem in problems)
        refused = Repaired(at, misses, scanned, None)
        self.record.write(
            name, _repair_record(refused), check=require_clean(redactor, name)
        )
        return refused


def _acted_through(ran: Sequence[Step]) -> set[str]:
    """The targets `ran`, the steps that ran, acted through."""
    return {
        step.target
        for step in ran
        if isinstance(step, Click | Fill | FillSecret | Select)
    }


def _ran_problems(
    patched: CompiledScript, in_use: CompiledScript, ran: Sequence[Step], at: At
) -> list[str]:
    """Why `patched` changes what ran, `ran` as `in_use` holds its steps. The
    validator reads only the committed script, so this also keeps an earlier
    repair's inserted steps that ran as they ran: what the record holds as run
    must be what ran. Every step has run by the time an assertion drifts, so
    its repair inserts none. A target a step acted through stays as that step
    found it, compared with `in_use`, so an earlier repair's binding may stay:
    otherwise the run's script would hold locators no step ran through."""
    kept = patched.steps[: len(ran)]
    if [step.seq for step in kept] != [step.seq for step in ran]:
        return ["/steps: a repair inserts no step before a step that already ran"]
    problems = [
        f"/steps/{index}: a repair rewrites no step that already ran"
        for index, (now, was) in enumerate(zip(kept, ran, strict=True))
        if now.model_dump(mode="json") != was.model_dump(mode="json")
    ]
    if isinstance(at, AssertionAt) and len(patched.steps) > len(ran):
        problems.append(
            "/steps: an assertion's repair inserts no step: every step has run"
        )
    problems.extend(
        (
            f"/targets/{pointer(name)}: a repair rebinds no target a step that "
            "already ran acted through"
        )
        for name in sorted(_acted_through(ran))
        if patched.targets[name] != in_use.targets[name]
    )
    return problems


def _repair_record(repaired: Repaired) -> dict[str, object]:
    at, script = repaired.at, repaired.script
    subject = (
        {"seq": at.step.seq} if isinstance(at, StepAt) else {"assertion": at.check.id}
    )
    return {
        **subject,
        "misses": list(repaired.misses),
        "problems": list(repaired.problems),
        "script": None
        if script is None
        else script.model_dump(mode="json", exclude_none=True),
    }
