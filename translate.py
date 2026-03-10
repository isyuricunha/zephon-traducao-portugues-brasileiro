"""
translate.py — Zephon game localization: English → Brazilian Portuguese
Uses the Mistral AI Agents API to translate XML language files.

Usage:
    python translate.py                          # translate all files
    python translate.py --file Colors.xml        # translate one file
    python translate.py --dry-run                # parse only, no API calls
    python translate.py --file Colors.xml --dry-run

Configuration:
    Copy .env.example to .env and fill MISTRAL_API_KEY + MISTRAL_AGENT_ID.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from mistralai import Mistral

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent
SOURCE_DIR = REPO_ROOT / "Original" / "Core" / "Languages" / "English"
TARGET_DIR = REPO_ROOT / "Data" / "Core" / "Languages" / "BrazilianPortuguese"
CHECKPOINT_FILE = REPO_ROOT / "translation_checkpoint.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(REPO_ROOT / "translate.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    api_key: str
    agent_id: str
    batch_size: int = 20
    max_retries: int = 3


def load_config() -> Config:
    """Load and validate configuration from .env file."""
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    agent_id = os.getenv("MISTRAL_AGENT_ID", "").strip()

    if not api_key:
        log.error("MISTRAL_API_KEY is not set. Copy .env.example to .env and fill in your key.")
        sys.exit(1)
    if not agent_id:
        log.error("MISTRAL_AGENT_ID is not set. Copy .env.example to .env and fill in your agent ID.")
        sys.exit(1)

    batch_size = int(os.getenv("BATCH_SIZE", "20"))
    max_retries = int(os.getenv("MAX_RETRIES", "3"))

    return Config(api_key=api_key, agent_id=agent_id, batch_size=batch_size, max_retries=max_retries)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Entry:
    """Represents a single <entry name="..." value="..."/> node."""

    name: str
    value: str
    translatable: bool = True
    translated_value: Optional[str] = None


# ---------------------------------------------------------------------------
# XML tag / placeholder detection helpers
# ---------------------------------------------------------------------------

# Match an entire value that is ONLY a <string name='...'/> reference.
_RE_PURE_STRING_REF = re.compile(r"^\s*<string\s+name='[^']*'\s*/>\s*$", re.IGNORECASE)

# Match values that contain no translatable letter at all (pure symbols, numbers, tags).
# Uses \p-style via a broad ASCII + Latin Extended range, avoiding broken Unicode spans.
_RE_NO_LETTERS = re.compile(r"^[^a-zA-Z\u00C0-\u024F]*$")

# Split a value string into alternating (text, xml_tag) segments.
# xml_tag is anything like <...> or </...>
_RE_XML_FRAGMENT = re.compile(r"(<[^>]+>)")


def is_skippable(value: str) -> bool:
    """Return True if the value needs no translation."""
    if _RE_PURE_STRING_REF.match(value):
        return True
    # Strip all XML tags to see if any letter content remains.
    text_only = _RE_XML_FRAGMENT.sub("", value)
    if _RE_NO_LETTERS.match(text_only):
        return True
    return False


def extract_text_segments(value: str) -> list[str]:
    """
    Split a value into a list of alternating plain-text and XML-tag tokens.
    Odd-indexed items are XML tags (must not be translated).
    Even-indexed items are plain text (translatable, may be empty string).
    """
    return _RE_XML_FRAGMENT.split(value)


def rebuild_value(original_segments: list[str], translated_texts: list[str]) -> str:
    """
    Reconstruct the full value string by merging translated text segments
    back together with the untouched XML tags.

    original_segments: result of extract_text_segments() on the original value.
    translated_texts: translated plain-text tokens (same count as even-index items).
    """
    result_parts: list[str] = []
    text_index = 0

    for i, segment in enumerate(original_segments):
        if i % 2 == 0:
            # Even index → plain text; replace with translation
            if text_index < len(translated_texts):
                result_parts.append(translated_texts[text_index])
                text_index += 1
            else:
                result_parts.append(segment)
        else:
            # Odd index → XML tag; keep exactly as-is
            result_parts.append(segment)

    return "".join(result_parts)


# ---------------------------------------------------------------------------
# Batch preparation — flatten entries into translatable text chunks
# ---------------------------------------------------------------------------


@dataclass
class TranslationUnit:
    """A single text chunk ready to be sent to the API."""

    entry_index: int       # index into the entry list
    segment_index: int     # which even-index segment within that entry's value
    text: str              # the raw English text to translate


def build_translation_units(entries: list[Entry]) -> list[TranslationUnit]:
    """
    For each translatable entry, extract all non-empty plain-text segments
    and build TranslationUnit objects.
    """
    units: list[TranslationUnit] = []
    for entry_idx, entry in enumerate(entries):
        if not entry.translatable:
            continue
        segments = extract_text_segments(entry.value)
        for seg_idx in range(0, len(segments), 2):  # even indices = plain text
            text = segments[seg_idx]
            if text.strip():  # skip empty
                units.append(TranslationUnit(
                    entry_index=entry_idx,
                    segment_index=seg_idx,
                    text=text,
                ))
    return units


def apply_translations(
    entries: list[Entry],
    units: list[TranslationUnit],
    translations: dict[tuple[int, int], str],
) -> None:
    """
    Write translated text back into each entry's translated_value.

    translations: maps (entry_index, segment_index) → translated text.
    """
    for entry_idx, entry in enumerate(entries):
        if not entry.translatable:
            entry.translated_value = entry.value
            continue

        segments = extract_text_segments(entry.value)
        rebuilt: list[str] = []
        for i, seg in enumerate(segments):
            if i % 2 == 0:
                key = (entry_idx, i)
                rebuilt.append(translations.get(key, seg))
            else:
                rebuilt.append(seg)
        entry.translated_value = "".join(rebuilt)


# ---------------------------------------------------------------------------
# Mistral Agent API call
# ---------------------------------------------------------------------------


def call_agent_batch(
    client: Mistral,
    agent_id: str,
    texts: list[str],
    max_retries: int,
    dry_run: bool,
) -> list[str]:
    """
    Send a batch of English strings to the Mistral Agent and return
    a list of translated strings (same length as input).

    On dry-run, returns the originals unchanged.
    On persistent failure, returns originals and logs error.
    """
    if dry_run:
        log.info("[DRY-RUN] Would translate %d strings.", len(texts))
        return texts

    prompt = (
        "Translate the following JSON array of game strings from English to Brazilian Portuguese (pt-BR). "
        "Return ONLY a valid JSON array of the same length, with each string translated. "
        "Do not include any explanation, markdown, or extra text — just the raw JSON array.\n\n"
        f"{json.dumps(texts, ensure_ascii=False)}"
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.agents.complete(
                agent_id=agent_id,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown code fences if the model adds them despite instructions.
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
                raw = raw.strip()

            translated = json.loads(raw)
            if not isinstance(translated, list):
                raise ValueError(f"Expected JSON array, got: {type(translated).__name__}")
            if len(translated) != len(texts):
                raise ValueError(
                    f"Length mismatch: sent {len(texts)} strings, got {len(translated)} back."
                )

            # Ensure all items are strings.
            return [str(t) for t in translated]

        except Exception as exc:
            wait = 2 ** attempt
            log.warning(
                "API attempt %d/%d failed: %s. Retrying in %ds…",
                attempt, max_retries, exc, wait,
            )
            if attempt < max_retries:
                time.sleep(wait)
            else:
                log.error(
                    "All %d attempts failed for batch of %d strings. "
                    "Falling back to original English values.",
                    max_retries, len(texts),
                )
                return texts

    return texts  # unreachable but satisfies type checker


# ---------------------------------------------------------------------------
# Checkpoint (resume support)
# ---------------------------------------------------------------------------


def load_checkpoint() -> dict:
    """Load checkpoint data; returns empty dict if file doesn't exist."""
    if CHECKPOINT_FILE.exists():
        try:
            with CHECKPOINT_FILE.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            log.warning("Could not load checkpoint: %s. Starting fresh.", exc)
    return {}


def save_checkpoint(checkpoint: dict) -> None:
    """Persist checkpoint data to disk."""
    try:
        with CHECKPOINT_FILE.open("w", encoding="utf-8") as fh:
            json.dump(checkpoint, fh, ensure_ascii=False, indent=2)
    except Exception as exc:
        log.warning("Could not save checkpoint: %s", exc)


# ---------------------------------------------------------------------------
# File I/O — custom line-based parser for non-standard XML
# ---------------------------------------------------------------------------
#
# The game's XML files are NOT valid XML 1.0: they contain raw "<tag>" markup
# inside attribute values (e.g. value="<icon .../> Some text"), which is
# illegal per the XML spec but accepted by the game's own parser.
# Using ElementTree raises ParseError on those files, so we parse line-by-line
# with regex, treating the file as plain text.
#

# Matches a complete <entry .../> line, capturing name and value.
# The value is everything between value=" and the final "/> at end of line.
_RE_ENTRY_LINE = re.compile(
    r'^(\s*)<entry\s+name="([^"]*?)"\s+value="(.*?)"\s*/>\s*$'
)

# Matches a complete <entry .../> where the value uses single-quoted attributes
# internally, so it may span patterns like value="...'...'".  The outer quotes
# are always double-quotes by convention in these files.


def parse_source_file(path: Path) -> tuple[list[Entry], list[str]]:
    """
    Parse a source XML file into Entry objects and the raw list of lines.

    Returns:
        entries: all parsed Entry objects in order.
        raw_lines: the original file lines (used verbatim for output).
    """
    raw_lines: list[str] = path.read_text(encoding="utf-8").splitlines(keepends=True)
    entries: list[Entry] = []

    for line in raw_lines:
        m = _RE_ENTRY_LINE.match(line)
        if not m:
            continue
        name = m.group(2)
        value = m.group(3)
        skippable = is_skippable(value)
        entries.append(Entry(name=name, value=value, translatable=not skippable))

    return entries, raw_lines


# ---------------------------------------------------------------------------
# XML writing — line-based replacement
# ---------------------------------------------------------------------------


def write_translated_file(
    entries: list[Entry],
    raw_lines: list[str],
    target_path: Path,
) -> None:
    """
    Write the translated file by substituting the value attribute on each
    <entry .../> line in the original raw_lines, preserving everything else
    (whitespace, comments, XML declaration, line endings) exactly.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Build name → translated value map.
    translation_map: dict[str, str] = {
        e.name: (e.translated_value if e.translated_value is not None else e.value)
        for e in entries
    }

    output_lines: list[str] = []
    for line in raw_lines:
        m = _RE_ENTRY_LINE.match(line)
        if m:
            indent = m.group(1)
            name = m.group(2)
            translated = translation_map.get(name)
            if translated is not None:
                # Reconstruct the line with the translated value, preserving
                # the original indentation and line ending.
                ending = "\r\n" if line.endswith("\r\n") else "\n"
                line = f'{indent}<entry name="{name}" value="{translated}"/>{ending}'
        output_lines.append(line)

    with target_path.open("w", encoding="utf-8", newline="") as fh:
        fh.writelines(output_lines)

    log.info("Written: %s", target_path.relative_to(REPO_ROOT))


# ---------------------------------------------------------------------------
# Per-file translation pipeline
# ---------------------------------------------------------------------------


def translate_file(
    source_path: Path,
    client: Mistral,
    config: Config,
    checkpoint: dict,
    dry_run: bool,
) -> None:
    """Translate a single XML file and write the output."""
    filename = source_path.name
    target_path = TARGET_DIR / filename

    log.info("=== Processing: %s ===", filename)

    entries, raw_lines = parse_source_file(source_path)
    translatable_count = sum(1 for e in entries if e.translatable)
    log.info(
        "  %d entries total, %d translatable, %d references/skipped.",
        len(entries),
        translatable_count,
        len(entries) - translatable_count,
    )

    if translatable_count == 0:
        log.info("  Nothing to translate. Copying as-is.")
        for entry in entries:
            entry.translated_value = entry.value
        write_translated_file(entries, raw_lines, target_path)
        checkpoint[filename] = "done"
        save_checkpoint(checkpoint)
        return

    # Build all translation units for this file.
    units = build_translation_units(entries)

    # Load any previously translated units from checkpoint.
    file_checkpoint: dict[str, str] = checkpoint.get(filename, {})
    if not isinstance(file_checkpoint, dict):
        file_checkpoint = {}

    translations: dict[tuple[int, int], str] = {}

    # Restore already-done translations from checkpoint.
    pending_units: list[TranslationUnit] = []
    for unit in units:
        key_str = f"{unit.entry_index}:{unit.segment_index}"
        if key_str in file_checkpoint:
            translations[(unit.entry_index, unit.segment_index)] = file_checkpoint[key_str]
        else:
            pending_units.append(unit)

    if file_checkpoint:
        log.info("  Resuming: %d/%d units already done from checkpoint.", len(units) - len(pending_units), len(units))

    # Translate pending units in batches.
    batch_size = config.batch_size
    for batch_start in range(0, len(pending_units), batch_size):
        batch = pending_units[batch_start : batch_start + batch_size]
        texts = [u.text for u in batch]

        log.info(
            "  Batch %d–%d of %d pending units…",
            batch_start + 1,
            min(batch_start + batch_size, len(pending_units)),
            len(pending_units),
        )

        translated_texts = call_agent_batch(
            client=client,
            agent_id=config.agent_id,
            texts=texts,
            max_retries=config.max_retries,
            dry_run=dry_run,
        )

        for unit, translated in zip(batch, translated_texts):
            key = (unit.entry_index, unit.segment_index)
            translations[key] = translated
            file_checkpoint[f"{unit.entry_index}:{unit.segment_index}"] = translated

        # Save checkpoint after every batch so progress is not lost.
        checkpoint[filename] = file_checkpoint
        save_checkpoint(checkpoint)

    # Apply translations back to entries.
    apply_translations(entries, units, translations)

    # Write the translated XML file.
    write_translated_file(entries, raw_lines, target_path)

    # Mark file as done in checkpoint.
    checkpoint[filename] = "done"
    save_checkpoint(checkpoint)
    log.info("  Done: %s", filename)


# ---------------------------------------------------------------------------
# Validation (post-translation sanity check)
# ---------------------------------------------------------------------------


def validate_output(source_path: Path, target_path: Path) -> bool:
    """
    Validate that the translated file:
    - Has the same number of <entry> lines.
    - Has the same set of `name` attributes.
    - Has no missing %N% placeholders compared to source.

    Returns True if all checks pass.
    """
    ok = True

    src_entries, _ = parse_source_file(source_path)
    tgt_entries, _ = parse_source_file(target_path)

    src_map = {e.name: e.value for e in src_entries}
    tgt_map = {e.name: (e.translated_value or e.value) for e in tgt_entries}

    # Check entry count.
    if len(src_map) != len(tgt_map):
        log.error(
            "VALIDATION FAIL: %s has %d entries but source has %d.",
            target_path.name, len(tgt_map), len(src_map),
        )
        ok = False

    # Check same names.
    missing_names = set(src_map) - set(tgt_map)
    if missing_names:
        log.error("VALIDATION FAIL: %s is missing entries: %s", target_path.name, missing_names)
        ok = False

    # Check placeholders are preserved.
    placeholder_re = re.compile(r"%%?\d+%%?")
    for name, src_val in src_map.items():
        if name not in tgt_map:
            continue
        tgt_val = tgt_map[name]
        src_placeholders = sorted(placeholder_re.findall(src_val))
        tgt_placeholders = sorted(placeholder_re.findall(tgt_val))
        if src_placeholders != tgt_placeholders:
            log.warning(
                "VALIDATION WARN: %s[%s] — placeholder mismatch. src=%s tgt=%s",
                target_path.name, name, src_placeholders, tgt_placeholders,
            )

    if ok:
        log.info("  Validation OK: %s", target_path.name)
    return ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate Zephon XML language files to pt-BR.")
    parser.add_argument(
        "--file",
        metavar="FILENAME",
        help="Translate only this file (e.g. Colors.xml). Default: all files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and batch files but do not call the API.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="After translation, validate output files against source.",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Ignore existing checkpoint and re-translate from scratch.",
    )
    args = parser.parse_args()

    # Load config (skipped on dry-run since no API call is made).
    if args.dry_run:
        config = Config(api_key="dry-run", agent_id="dry-run")
    else:
        config = load_config()

    client = Mistral(api_key=config.api_key)

    # Load or reset checkpoint.
    if args.reset_checkpoint and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        log.info("Checkpoint reset.")
    checkpoint = load_checkpoint()

    # Gather source files.
    if args.file:
        source_files = [SOURCE_DIR / args.file]
        if not source_files[0].exists():
            log.error("File not found: %s", source_files[0])
            sys.exit(1)
    else:
        source_files = sorted(SOURCE_DIR.glob("*.xml"))

    log.info("Source directory : %s", SOURCE_DIR)
    log.info("Target directory : %s", TARGET_DIR)
    log.info("Files to process : %d", len(source_files))
    if args.dry_run:
        log.info("Mode             : DRY-RUN (no API calls)")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    for source_path in source_files:
        # Skip files already fully done (unless --reset-checkpoint used).
        if checkpoint.get(source_path.name) == "done":
            log.info("Skipping (already done): %s", source_path.name)
            continue
        try:
            translate_file(
                source_path=source_path,
                client=client,
                config=config,
                checkpoint=checkpoint,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            log.error("ERROR processing %s: %s", source_path.name, exc, exc_info=True)
            errors.append(source_path.name)

    # Optional validation pass.
    if args.validate and not args.dry_run:
        log.info("=== Validation pass ===")
        for source_path in source_files:
            target_path = TARGET_DIR / source_path.name
            if target_path.exists():
                validate_output(source_path, target_path)

    if errors:
        log.error("Finished with errors in: %s", ", ".join(errors))
        sys.exit(1)
    else:
        log.info("All files processed successfully.")


if __name__ == "__main__":
    main()
