"""The OCR check on screenshots (#250, ADR-0035): in a run that binds a test
secret, a capture whose image's text spells a bound value, live or
rendered, is withheld, and the record saves nothing."""

import asyncio
import html
from base64 import b64encode
from collections.abc import Iterator
from pathlib import Path

import pytest
from aqa_core.project import SecretDestination
from aqa_runner import ocr, screenshots
from aqa_runner.bound_secrets import BoundSecret
from aqa_runner.browser_session import BrowserSession
from aqa_runner.redaction import NO_SECRETS, Redactor
from aqa_runner.run_record import RunRecord
from aqa_runner.screenshots import ScreenshotWithheldError, take_screenshot
from pydantic import SecretStr

from packages.runner.tests.document_fixtures import Sites, browsing, serving_sites
from packages.runner.tests.screenshot_pages import (
    RED_VALUE,
    UNBOUND,
    bound,
    red_filled,
    serve_cases,
)

# Draws `text` into an image on the page, as a page can once the value is
# in: a render carries a data: image, and no scrub reads its pixels.
DRAW = """(text) => {
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 80;
    const context = canvas.getContext("2d");
    context.fillStyle = "#fff";
    context.fillRect(0, 0, 640, 80);
    context.fillStyle = "#000";
    context.font = "40px sans-serif";
    context.fillText(text, 10, 55);
    const image = Object.assign(new Image(), {src: canvas.toDataURL()});
    image.style.cssText = "position:absolute;left:0;top:200px";
    document.body.append(image);
    return image.decode();
}"""

# Why a capture whose image's text spells a bound value is withheld.
SPELLED = "the image's text spells a bound value"


@pytest.fixture
def sites(monkeypatch: pytest.MonkeyPatch) -> Iterator[Sites]:
    with serving_sites(monkeypatch) as served:
        yield served


async def saved(session: BrowserSession, record: RunRecord) -> tuple[str, list[str]]:
    """The session's screenshot written to `record`, or why it is withheld,
    and the files the record then holds."""
    try:
        record.write_screenshot("shots/0.png", await take_screenshot(session))
    except ScreenshotWithheldError as withheld:
        outcome = str(withheld)
    else:
        outcome = "saved"
    files = sorted(
        path.relative_to(record.path).as_posix()
        for path in record.path.rglob("*")
        if path.is_file()
    )
    return outcome, files


def rendered(sites: Sites, tmp_path: Path, text: str) -> tuple[str, list[str]]:
    """What `saved` reports once the red value is filled, so a capture is a
    render, and the page has drawn `text` into an image."""

    async def scenario() -> tuple[str, list[str]]:
        async with browsing(sites, redactor=Redactor(bound(sites.app))) as session:
            await serve_cases(session.page, sites.other)
            record = RunRecord.create(tmp_path).redacting(session.redactor)
            await red_filled(session, sites)
            await session.page.evaluate(DRAW, text)
            return await saved(session, record)

    return asyncio.run(scenario())


def reflected(
    sites: Sites,
    tmp_path: Path,
    text: str,
    redactor: Redactor | None = None,
    *,
    font: str = "40px sans-serif",
) -> tuple[str, list[str]]:
    """What `shown` reports of a page that shows `text` in `font`, as an
    app's server can show a bound value before any fill."""
    body = f"<p style='font:{font};margin:20px'>{html.escape(text)}</p>"
    return shown(sites, tmp_path, body, redactor)


def shown(
    sites: Sites, tmp_path: Path, body: str, redactor: Redactor | None = None
) -> tuple[str, list[str]]:
    """What `saved` reports of a page of `body`, so the capture is the live
    page's, in a session that scans with `redactor` (by default, of the
    test's bound secrets)."""
    page = f"<!doctype html><meta charset=utf-8>{body}"

    async def scenario() -> tuple[str, list[str]]:
        scan = Redactor(bound(sites.app)) if redactor is None else redactor
        async with browsing(sites, redactor=scan) as session:
            await session.page.route(
                "**/reflected",
                lambda route: route.fulfill(
                    body=page, content_type="text/html; charset=utf-8"
                ),
            )
            record = RunRecord.create(tmp_path).redacting(session.redactor)
            await session.navigate(f"{sites.app}/reflected")
            return await saved(session, record)

    return asyncio.run(scenario())


def test_a_render_whose_image_shows_a_bound_value_is_dropped(
    sites: Sites, tmp_path: Path
) -> None:
    assert rendered(sites, tmp_path, RED_VALUE) == (SPELLED, [])


def test_a_render_whose_image_shows_an_unbound_value_is_kept(
    sites: Sites, tmp_path: Path
) -> None:
    assert rendered(sites, tmp_path, UNBOUND) == ("saved", ["shots/0.png"])


def test_a_live_page_that_shows_a_bound_value_before_any_fill_is_dropped(
    sites: Sites, tmp_path: Path
) -> None:
    assert reflected(sites, tmp_path, RED_VALUE) == (SPELLED, [])


def test_a_live_page_that_shows_an_unbound_value_is_kept(
    sites: Sites, tmp_path: Path
) -> None:
    assert reflected(sites, tmp_path, UNBOUND) == ("saved", ["shots/0.png"])


def test_a_base64_spelling_the_image_shows_is_dropped(
    sites: Sites, tmp_path: Path
) -> None:
    spelled = b64encode(RED_VALUE.encode()).decode()

    assert reflected(sites, tmp_path, spelled) == (SPELLED, [])


# Split as a page may show a key, in groups: OCR reads the spaces too, and
# can add some of its own.
def test_a_value_the_image_shows_split_by_spaces_is_dropped(
    sites: Sites, tmp_path: Path
) -> None:
    grouped = " ".join(RED_VALUE[start : start + 4] for start in range(0, 11, 4))

    assert reflected(sites, tmp_path, grouped) == (SPELLED, [])


def phrased(sites: Sites) -> Redactor:
    """A scan of one bound value with spaces in it."""
    return Redactor(
        [
            BoundSecret(
                "PHRASE",
                SecretStr("fake pass phrase"),
                SecretDestination((sites.app,), "password"),
            )
        ]
    )


# Read as OCR read it, too: with its spaces taken out, a value that has
# some isn't the value.
def test_a_value_with_spaces_in_it_the_image_shows_is_dropped(
    sites: Sites, tmp_path: Path
) -> None:
    assert reflected(sites, tmp_path, "fake pass phrase", phrased(sites)) == (
        SPELLED,
        [],
    )


# Its word spaces and the page's own, which match neither the value as
# written nor the value with OCR's spaces taken out of the text alone.
def test_a_value_with_spaces_in_it_the_image_shows_in_groups_is_dropped(
    sites: Sites, tmp_path: Path
) -> None:
    assert reflected(sites, tmp_path, "f a k e pass phrase", phrased(sites)) == (
        SPELLED,
        [],
    )


def test_a_run_that_binds_nothing_runs_no_ocr(
    sites: Sites, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    read: list[bytes] = []

    def reading(png: bytes, _redactor: Redactor) -> bool:
        read.append(png)
        return True

    monkeypatch.setattr(ocr, "spells_a_value", reading)

    assert reflected(sites, tmp_path, RED_VALUE, NO_SECRETS) == (
        "saved",
        ["shots/0.png"],
    )
    assert read == []


def test_ocr_past_its_bound_withholds_it(
    sites: Sites, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(screenshots, "OCR_SECONDS", 0.05)

    assert reflected(sites, tmp_path, UNBOUND) == (
        "the image's OCR took over 0.05 s",
        [],
    )


# A page of small text, more than OCR reads in its bound on any host.
DENSE = " ".join(f"word{place} lorem ipsum" for place in range(4000))


def test_text_too_long_to_read_in_the_bound_withholds_it(
    sites: Sites, tmp_path: Path
) -> None:
    assert reflected(sites, tmp_path, DENSE, font="10px sans-serif") == (
        "the image has more text than OCR reads in its bound",
        [],
    )


# 144 numbers far apart. Each is read at the recogniser's shortest, 320/48,
# so the image's text is 960 long in any host's fonts.
SPREAD = (
    "<div style='display:grid;grid-template-columns:repeat(12,1fr);"
    "grid-auto-rows:64px;font:16px sans-serif'>"
    + "".join(f"<div>{place:03}</div>" for place in range(144))
    + "</div>"
)


def test_an_image_just_under_the_most_text_ocr_reads_is_read_in_its_bound(
    sites: Sites, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at_the_cap = shown(sites, tmp_path, SPREAD)
    monkeypatch.setattr(ocr, "TEXT_CAP", ocr.TEXT_CAP * 0.95)

    assert (at_the_cap, shown(sites, tmp_path, SPREAD)) == (
        ("saved", ["shots/0.png"]),
        ("the image has more text than OCR reads in its bound", []),
    )


def test_a_missing_model_file_saves_nothing_and_downloads_nothing(
    sites: Sites, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(ocr, "MODELS", models)

    with pytest.raises(FileNotFoundError):
        reflected(sites, tmp_path, UNBOUND)

    # The run record's own .gitignore: no screenshot, and no model file.
    assert [path.name for path in tmp_path.rglob("*") if path.is_file()] == [
        ".gitignore"
    ]
