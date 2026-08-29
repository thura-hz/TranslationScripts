#!/usr/bin/env python3
"""
translate_srt.py — Semi-interactive English -> Burmese subtitle translator
using the Gemini API (google-genai SDK).

Workflow per chunk of subtitle cues:
  1. Draft translation (Gemini)
  2. Self-critique / refine pass (Gemini reviews its own draft)
  3. Show you the result -> accept all / edit specific lines / re-refine / quit
  4. Progress is saved after every chunk, so you can stop and resume any time.

Usage:
    export GEMINI_API_KEY="your-key-here"
    pip install google-genai srt --break-system-packages
    python translate_srt.py input.srt output.srt

Resuming:
    Just run the same command again with the same input/output pair —
    it picks up from the sidecar progress file automatically.
"""

import argparse
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path

import srt
import pysubs2
from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash"  # current GA flash model as of mid-2026; override with --model if this changes again
CHUNK_SIZE = 40              # subtitle cues per chunk
CONTEXT_TAIL = 4             # how many previous translated lines to show for continuity
FLAG_SIMILARITY_THRESHOLD = 0.55  # below this draft<->refined similarity, flag for human review


def significant_change(draft: str, refined: str) -> bool:
    """True if the refine pass changed the line enough to be worth a human look."""
    if not draft or not refined:
        return True
    ratio = difflib.SequenceMatcher(None, draft, refined).ratio()
    return ratio < FLAG_SIMILARITY_THRESHOLD


BANNED_CHARS = "၊။!?"


def clean_text(text: str) -> str:
    """Safety net: strip banned punctuation regardless of what the model outputs,
    then tidy up any resulting double spaces / trailing space left behind."""
    if not text:
        return text
    for ch in BANNED_CHARS:
        text = text.replace(ch, "")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def strip_stage_directions(text: str) -> str:
    """Remove parenthetical/bracketed sound-and-action cues like '(laughs)',
    '(door creaks)', '[music playing]' from English subtitle text before
    translation, since these are directions, not dialogue to be translated."""
    if not text:
        return text
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

KEYS_CONFIG_PATH = Path.home() / ".config" / "translate_srt" / "keys.json"


def load_keys_config() -> dict:
    if KEYS_CONFIG_PATH.exists():
        return json.loads(KEYS_CONFIG_PATH.read_text(encoding="utf-8"))
    return {"keys": {}, "default": None}


def save_keys_config(cfg: dict):
    KEYS_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    KEYS_CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    os.chmod(KEYS_CONFIG_PATH, 0o600)  # keys are sensitive, keep it user-readable only


def get_client(profile: str = None) -> genai.Client:
    cfg = load_keys_config()

    api_key = None
    if profile:
        api_key = cfg["keys"].get(profile)
        if not api_key:
            sys.exit(f"No saved key named '{profile}'. Add one with: --save-key {profile} YOUR_KEY")
    elif cfg.get("default") and cfg["keys"].get(cfg["default"]):
        api_key = cfg["keys"][cfg["default"]]
    else:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        sys.exit(
            "No API key found. Either:\n"
            "  1) Save one:   python translate_srt.py --save-key main YOUR_KEY\n"
            "  2) Or set env: set -x GEMINI_API_KEY \"your-key\"   (fish shell)\n"
        )
    return genai.Client(api_key=api_key)


class QuotaExhaustedError(Exception):
    """Raised when the DAILY request quota is hit — retrying won't help until reset."""
    pass


class PersistentEmptyResponseError(Exception):
    """Raised when the model keeps returning empty responses even after retries —
    usually means the prompt content tripped a safety filter, not a transient fluke."""
    pass


def call_gemini(client: genai.Client, prompt: str, max_retries: int = 6) -> str:
    """Call Gemini with JSON output mode. Distinguishes between:
    - per-minute/per-token limits (transient, worth backing off and retrying)
    - per-day limits (won't clear until midnight Pacific Time, no point retrying)
    Adds a small pacing delay after every successful call to avoid bursting RPM."""
    delay = 3
    use_thinking_config = True
    empty_response_attempts = 0
    for attempt in range(max_retries):
        try:
            config_kwargs = dict(
                response_mime_type="application/json",
                temperature=0.4,
                max_output_tokens=65536,
            )
            if use_thinking_config:
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            if not resp.text:
                finish_reason = None
                block_reason = None
                try:
                    finish_reason = resp.candidates[0].finish_reason
                except Exception:
                    pass
                try:
                    block_reason = resp.prompt_feedback.block_reason
                except Exception:
                    pass
                raise RuntimeError(f"EMPTY_RESPONSE (finish_reason={finish_reason}, block_reason={block_reason})")
            time.sleep(4)  # simple pacing so we don't burst past RPM even on fast responses
            return resp.text
        except Exception as e:
            msg = str(e)
            if use_thinking_config and ("thinking_level" in msg or "thinking_budget" in msg):
                # older/non-3.x models don't support thinking_level — drop it and retry immediately
                use_thinking_config = False
                print("  [model doesn't support thinking_level, retrying without it...]")
                continue
            if "EMPTY_RESPONSE" in msg:
                empty_response_attempts += 1
                if empty_response_attempts >= 3:
                    # 3 empty responses in a row on the same content is very unlikely to be
                    # random — almost certainly a safety filter on the content itself. Fail
                    # fast so the caller can bisect and isolate the actual problem line(s)
                    # instead of burning the full retry budget on a doomed request.
                    raise PersistentEmptyResponseError(msg)
                print(f"  [empty response from model ({msg}), retrying in {delay}s...]")
                time.sleep(delay)
                delay = min(delay * 2, 90)
                continue
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                if "PerDay" in msg or "per_day" in msg.lower() or "daily" in msg.lower():
                    raise QuotaExhaustedError(
                        "Daily request quota (RPD) exhausted. This resets at midnight Pacific "
                        "Time — no amount of waiting right now will fix it. Progress so far is "
                        "saved; re-run the same command after the reset to resume."
                    )
                print(f"  [rate limited (per-minute/token), waiting {delay}s...]")
                time.sleep(delay)
                delay = min(delay * 2, 90)
                continue
            raise
    raise QuotaExhaustedError(
        "Repeated rate-limit errors that didn't clear after several backoff attempts. "
        "This usually means the per-minute window is unusually congested, or the daily "
        "quota is exhausted but wasn't clearly labeled. Progress so far is saved — wait "
        "a few minutes (or until midnight Pacific Time if it's the daily cap) and re-run "
        "the same command to resume."
    )


def parse_json_array(text: str):
    """Gemini sometimes wraps JSON in fences even in JSON mode; strip defensively."""
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


def parse_json_array_lenient(text: str):
    """Like parse_json_array, but if the response got truncated (hit the token
    limit mid-object) this salvages every complete {...} object it can find
    instead of failing the whole batch. Returns whatever was recoverable."""
    if not text:
        return []
    try:
        return parse_json_array(text)
    except Exception:
        pass
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    objects = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start:i + 1]
                try:
                    objects.append(json.loads(candidate))
                except Exception:
                    pass
                start = None
    return objects


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_translate_prompt(items, glossary, context_lines):
    numbered = "\n".join(f"{it['id']} | {it['text']}" for it in items)
    glossary_txt = json.dumps(glossary, ensure_ascii=False, indent=2) if glossary else "(none yet)"
    context_txt = "\n".join(context_lines) if context_lines else "(this is the start of the video)"
    return f"""You are a professional subtitle translator translating English video subtitles
into natural, natively-fluent Burmese (Myanmar language / မြန်မာဘာသာ), for general audience viewing.

Rules:
- Translate MEANING and tone naturally. Do not translate word-for-word / do not sound like machine translation.
- TONE / ACCENT: Use everyday SPOKEN Burmese (ပြောဆိုသုံးနှုန်းသော ဘာသာစကား), the way people actually
  talk out loud, NOT written/literary Burmese (စာပေဆန်သော / ရေးသားသုံးနှုန်းသော ဘာသာစကား) and not
  textbook or news-broadcast style. Prefer colloquial word choices, contractions, and natural sentence
  endings a native speaker would casually say, over formal literary vocabulary or stiff grammatical
  constructions, unless the English line itself is clearly formal (e.g. a speech, a stranger, a boss to
  an employee) — in which case match that formality instead.
- Keep each line short enough to read comfortably as a subtitle (avoid padding or explanations).
- Use the glossary below consistently for names and recurring terms.
- Do not re-translate the CONTEXT lines, they are only for continuity.
- Preserve informal/formal register matching the English tone.
- PUNCTUATION: Do NOT use the Burmese punctuation marks ၊ or ။ anywhere. Do NOT use "!" or "?" either.
  End lines with no punctuation mark at all, relying on line breaks and phrasing/word choice alone
  to convey emphasis or questions naturally.
- ADDRESS TERMS (2nd person "you" / self-reference "I"): Burmese rarely uses a generic "you" between
  people who know each other — using generic pronouns for family or close relationships sounds cold
  or rude. Instead use RELATIONSHIP-BASED kinship/role terms for both addressing the other person and
  often for self-reference, based on who is speaking to whom. Examples (adapt naturally to context):
    • son speaking to mother: address her "မေမေ", refer to self as "သား"
    • mother speaking to son: address him "သား", refer to self as "မေမေ"
    • daughter speaking to mother: address her "မေမေ", refer to self as "သမီး"
    • child speaking to father: address him "ဖေဖေ", refer to self as "သား"/"သမီး"
    • younger sibling to elder brother/sister: address them "အကို"/"အစ်မ", self as "ညီ"/"ညီမ" or by name
    • elder sibling to younger: address by name or "ညီ"/"ညီမ", self as "အကို"/"အမ"
    • student to teacher, younger person to older non-family adult: use "ဆရာ"/"ဆရာမ", "ဦးလေး"/"အန်တီ"
      or similar respectful role terms rather than a generic pronoun.
  Reserve the gendered pronoun pair "ရှင်" (female speaking to male) / "မင်း" (male speaking to female)
  ONLY for relationships with no closer specific term — e.g. adult strangers, casual peers/acquaintances,
  or romantic partners — NOT for family members or any relationship with a clear kinship/hierarchical term.
  Use the glossary's tracked character relationships to keep these terms fully consistent for the same
  character pair throughout the whole video.

GLOSSARY (includes known character genders/relationships where established):
{glossary_txt}

CONTEXT (previous translated lines, for continuity only):
{context_txt}

SUBTITLES TO TRANSLATE (id | English text):
{numbered}

Output ONLY a valid JSON array, one object per input line, same order, same ids:
[{{"id": <int>, "text": "<Burmese translation>"}}, ...]
"""


def build_refine_prompt(items, glossary):
    rows = "\n".join(f"{it['id']} | EN: {it['english']} | DRAFT: {it['draft']}" for it in items)
    glossary_txt = json.dumps(glossary, ensure_ascii=False, indent=2) if glossary else "(none yet)"
    return f"""You are reviewing your own Burmese subtitle translation before final delivery.

For each item, compare the English source and the draft Burmese translation.
If the draft is already natural, accurate, and appropriately concise, keep it unchanged.
If it sounds unnatural, too literal, awkward, stiff, or overly formal/literary, rewrite it so it
reads like something a native Burmese speaker would warmly and naturally SAY OUT LOUD in this
context — everyday spoken Burmese, not written/literary Burmese (စာပေဆန်သော) and not textbook or
news-broadcast style — unless the scene itself calls for formality.

PUNCTUATION: The final text must NOT contain ၊ ။ ! or ? anywhere. If the draft has any of these,
remove them and rephrase the ending naturally instead of just deleting the mark mid-sentence.

ADDRESS TERMS: Check "you"/"I" against this rule — family and close/hierarchical relationships should
use kinship or role terms (e.g. son↔mother: "သား"/"မေမေ", child↔father: "သား"or"သမီး"/"ဖေဖေ",
sibling terms, "ဆရာ"/"ဆရာမ" for teacher, etc.), NOT a generic pronoun, since generic "you" sounds
rude between people who know each other. Only use the gendered pair "ရှင်" (female→male) / "မင်း"
(male→female) for relationships with no closer specific term (strangers, casual peers, romantic
partners). Use the glossary's known character relationships to fix any line that used a generic
pronoun where a kinship/role term was called for, or that was inconsistent with how the same pair
addressed each other earlier.

GLOSSARY (includes known character genders/relationships where established):
{glossary_txt}

ITEMS (id | EN | DRAFT):
{rows}

Output ONLY a valid JSON array:
[{{"id": <int>, "text": "<final Burmese text>", "changed": true/false}}, ...]
"""


def build_glossary_prompt(existing_glossary, pairs):
    glossary_txt = json.dumps(existing_glossary, ensure_ascii=False, indent=2) if existing_glossary else "{}"
    pairs_txt = "\n".join(f"EN: {p['english']}  ->  MY: {p['burmese']}" for p in pairs)
    return f"""From these English subtitle lines and their Burmese translations, extract/update a short
glossary of proper names and recurring terms that should stay CONSISTENT for the rest of this video.

Also track character GENDER and RELATIONSHIPS whenever identifiable from dialogue or context (names,
titles, how characters refer to or address each other, family/social roles mentioned) — this is used
to pick the correct relationship-based address term ("you"/"I") for each character pair (e.g. son
calling his mother "မေမေ" and referring to himself as "သား", rather than a generic pronoun). For any
named character with known gender and/or relationships, use a key formatted as:
  "CharacterName [gender: male/female, relation: e.g. son of X / mother of Y / husband of Z / teacher of W]"
mapped to their Burmese name. Only include the relation part when it's actually identifiable.

Existing glossary (extend or correct, keep entries that are still valid):
{glossary_txt}

New material:
{pairs_txt}

Output ONLY a valid JSON object mapping English term (or "Name [gender: ..., relation: ...]" for
characters with known info) -> chosen Burmese translation. Keep it to the most important recurring
terms and characters (max ~20 entries total).
"""


# ---------------------------------------------------------------------------
# Progress persistence
# ---------------------------------------------------------------------------

def progress_path(output_path: str) -> Path:
    return Path(output_path).with_suffix(".progress.json")


def load_progress(output_path: str):
    p = progress_path(output_path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"glossary": {}, "translations": {}}


def save_progress(output_path: str, state: dict):
    progress_path(output_path).write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class Cue:
    """Uniform interface used by the rest of the pipeline, regardless of source format."""
    __slots__ = ("index", "content")

    def __init__(self, index, content):
        self.index = index
        self.content = content


def detect_encoding(path: str) -> str:
    """Subtitle files from various sources are frequently NOT UTF-8. Windows-1252 is
    overwhelmingly the most common legacy encoding for subtitle files specifically
    (it's what Windows text editors/tools have defaulted to for decades), so it's
    tried right after UTF-8 — before falling back to generic statistical detection,
    which tends to be unreliable on short, subtitle-sized text samples."""
    raw = Path(path).read_bytes()
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        raw.decode("cp1252")
        return "cp1252"
    except UnicodeDecodeError:
        pass
    try:
        import charset_normalizer
        result = charset_normalizer.from_bytes(raw).best()
        if result and result.encoding:
            return result.encoding
    except Exception:
        pass
    return "latin-1"  # never fails to decode — last resort


def load_subtitle_file(path: str):
    """Returns (fmt, raw_data, cues).
    - fmt: 'srt' or 'ass'
    - raw_data: the original parsed object needed to reconstruct the file on write
                (list of srt.Subtitle, or a pysubs2.SSAFile)
    - cues: List[Cue] — uniform (index, content) view used by the rest of the script
    """
    encoding = detect_encoding(path)
    if encoding.lower() not in ("utf-8", "utf8"):
        print(f"Note: {Path(path).name} isn't UTF-8, reading it as {encoding}.")

    ext = Path(path).suffix.lower()
    if ext in (".ass", ".ssa"):
        subs = pysubs2.load(path, encoding=encoding)
        # only Dialogue lines are ever translated; Comment lines pass through untouched.
        # Position in the event list is used as the stable id (no numeric field exists
        # in ASS the way SRT has one).
        cues = [Cue(i, ev.plaintext) for i, ev in enumerate(subs) if ev.type == "Dialogue"]
        return "ass", subs, cues
    else:
        subs = list(srt.parse(Path(path).read_text(encoding=encoding)))
        cues = [Cue(s.index, s.content) for s in subs]
        return "srt", subs, cues


def write_subtitle_file(fmt: str, raw_data, translations: dict, output_path: str):
    if fmt == "ass":
        subs = raw_data
        for i, ev in enumerate(subs):
            if ev.type != "Dialogue":
                continue
            text = translations.get(str(i))
            if text is not None:
                ev.plaintext = text
        subs.save(output_path)
    else:
        out_subs = []
        for sub in raw_data:
            text = translations.get(str(sub.index), sub.content)
            out_subs.append(srt.Subtitle(index=sub.index, start=sub.start, end=sub.end, content=text))
        # reindex=False: keep each cue's ORIGINAL index number. srt.compose() defaults to
        # reindex=True, which silently renumbers everything sequentially by position and
        # breaks the label<->timestamp correspondence whenever the source file has any
        # gaps or non-sequential numbering.
        Path(output_path).write_text(srt.compose(out_subs, reindex=False), encoding="utf-8")


def translate_with_bisection(client, items, glossary, context_lines, depth=0):
    """Draft-translate a list of items. If the model persistently returns empty
    responses (safety-filtered content), split the batch in half and retry each
    half separately — narrowing down to the actual problem line(s) instead of
    losing the whole batch. Returns (draft_by_id dict, list of permanently
    skipped ids that couldn't be translated at all)."""
    if not items:
        return {}, []

    prompt = build_translate_prompt(items, glossary, context_lines)
    try:
        raw = call_gemini(client, prompt)
    except PersistentEmptyResponseError as e:
        if len(items) == 1:
            print(f"  [cue {items[0]['id']} appears to be blocked by content filtering ({e}) — "
                  f"leaving it untranslated, will need manual translation for this line]")
            return {}, [items[0]["id"]]
        mid = len(items) // 2
        print(f"  [batch of {len(items)} lines got persistent empty responses, "
              f"splitting in half to isolate the problem line...]")
        left_draft, left_skipped = translate_with_bisection(client, items[:mid], glossary, context_lines, depth + 1)
        right_draft, right_skipped = translate_with_bisection(client, items[mid:], glossary, context_lines, depth + 1)
        return {**left_draft, **right_draft}, left_skipped + right_skipped

    data = parse_json_array_lenient(raw)
    draft_by_id = {d["id"]: clean_text(d["text"]) for d in data if "id" in d and "text" in d}
    missing_ids = [it["id"] for it in items if it["id"] not in draft_by_id]
    if missing_ids and len(items) > 1:
        print(f"  [{len(missing_ids)} line(s) missing from draft response (likely truncated), retrying just those...]")
        missing_items = [it for it in items if it["id"] in missing_ids]
        retry_draft, retry_skipped = translate_with_bisection(client, missing_items, glossary, context_lines, depth + 1)
        draft_by_id.update(retry_draft)
        return draft_by_id, retry_skipped
    return draft_by_id, [i for i in missing_ids if i not in draft_by_id]


# ---------------------------------------------------------------------------
# Interactive review
# ---------------------------------------------------------------------------

def review_flagged(flagged_items, refined_by_id, draft_by_id, state):
    """
    Only shown for lines the refine pass changed significantly.
    flagged_items: [{'id','english'}]
    Returns updated {id: text} for the flagged ids, or None to signal quit.
    """
    ids = [it["id"] for it in flagged_items]

    while True:
        print("\n" + "-" * 70)
        print(f"{len(ids)} line(s) changed significantly during refine — quick check:")
        for it in flagged_items:
            i = it["id"]
            print(f"[{i}] EN:     {it['english']}")
            print(f"     DRAFT:  {draft_by_id.get(i, '')}")
            print(f"     FINAL:  {refined_by_id.get(i, '')}")
        print("-" * 70)
        choice = input(
            "[a]ccept these  [e]dit <id>  [r]e-refine w/ note  [q]uit&save  > "
        ).strip()

        if choice.lower() == "a":
            return {i: refined_by_id[i] for i in ids}
        elif choice.lower() == "q":
            return None
        elif choice.lower().startswith("e"):
            parts = choice.split()
            target_id = int(parts[1]) if len(parts) > 1 else int(input("Edit which id? "))
            new_text = input(f"New Burmese text for [{target_id}]: ").strip()
            if new_text:
                refined_by_id[target_id] = clean_text(new_text)
        elif choice.lower().startswith("r"):
            note = input("What should the reviewer fix/focus on? ").strip()
            client = state["client"]
            items_for_refine = [
                {"id": it["id"], "english": it["english"], "draft": refined_by_id[it["id"]]}
                for it in flagged_items
            ]
            prompt = build_refine_prompt(items_for_refine, state["glossary"])
            prompt += f"\n\nADDITIONAL INSTRUCTION FROM TRANSLATOR: {note}\n"
            raw = call_gemini(client, prompt)
            try:
                data = parse_json_array(raw)
                for d in data:
                    refined_by_id[d["id"]] = clean_text(d["text"])
            except Exception as e:
                print(f"  [could not parse refine response: {e}] — keeping previous version")
        else:
            print("  (unrecognized option)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    global FLAG_SIMILARITY_THRESHOLD, MODEL
    ap = argparse.ArgumentParser(description="Translate an English .srt or .ass/.ssa subtitle file to Burmese via Gemini, semi-interactively.")
    ap.add_argument("input_srt", nargs="?")
    ap.add_argument("output_srt", nargs="?")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--flag-threshold", type=float, default=FLAG_SIMILARITY_THRESHOLD,
                     help="Lower = stricter (more lines flagged for review). Default 0.55.")
    ap.add_argument("--model", default=MODEL,
                     help=f"Gemini model to use (default: {MODEL}). Override if Google renames/retires it again, "
                          "e.g. --model gemini-flash-latest to always use whatever is current.")
    ap.add_argument("--clean-existing", action="store_true",
                     help="Just strip banned punctuation (၊ ။ ! ?) from already-saved progress and exit, no new API calls.")
    ap.add_argument("--redo-range", nargs=2, type=int, metavar=("START_ID", "END_ID"),
                     help="Force re-translation of cue ids START_ID..END_ID (inclusive) even if already done, "
                          "e.g. --redo-range 1 40 to redo the first chunk with updated prompt rules.")
    ap.add_argument("--save-key", nargs=2, metavar=("PROFILE", "KEY"),
                     help="Save an API key under a name, e.g. --save-key main AIza... or --save-key alt AIza...")
    ap.add_argument("--set-default-key", metavar="PROFILE",
                     help="Make a saved profile the default so you never have to specify --use-key again.")
    ap.add_argument("--use-key", metavar="PROFILE",
                     help="Use a saved profile's key for just this run, without changing the default.")
    ap.add_argument("--list-keys", action="store_true", help="List saved key profile names (not the keys themselves).")
    args = ap.parse_args()
    FLAG_SIMILARITY_THRESHOLD = args.flag_threshold
    MODEL = args.model

    # --- key management commands (no srt file needed) ---
    if args.save_key:
        profile, key = args.save_key
        cfg = load_keys_config()
        cfg["keys"][profile] = key
        if cfg.get("default") is None:
            cfg["default"] = profile  # first saved key becomes default automatically
        save_keys_config(cfg)
        print(f"Saved key profile '{profile}'." + (" Set as default." if cfg["default"] == profile else ""))
        return

    if args.set_default_key:
        cfg = load_keys_config()
        if args.set_default_key not in cfg["keys"]:
            sys.exit(f"No saved profile named '{args.set_default_key}'. Saved profiles: {list(cfg['keys'].keys())}")
        cfg["default"] = args.set_default_key
        save_keys_config(cfg)
        print(f"Default key profile set to '{args.set_default_key}'.")
        return

    if args.list_keys:
        cfg = load_keys_config()
        if not cfg["keys"]:
            print("No saved key profiles yet. Add one with: --save-key <name> <key>")
        else:
            for name in cfg["keys"]:
                marker = " (default)" if name == cfg.get("default") else ""
                print(f"  {name}{marker}")
        return

    if not args.input_srt or not args.output_srt:
        sys.exit("Usage: python translate_srt.py input.srt output.srt [options]\n"
                  "(Or use --save-key / --list-keys / --set-default-key for API key management.)")

    fmt, raw_data, subs = load_subtitle_file(args.input_srt)
    expected_ext = {"srt": ".srt", "ass": ".ass"}[fmt]
    out_ext = Path(args.output_srt).suffix.lower()
    if fmt == "ass" and out_ext not in (".ass", ".ssa"):
        print(f"Warning: input is {fmt.upper()} but output filename doesn't end in .ass/.ssa "
              f"— the file will still be written in {fmt.upper()} format regardless of its name.")
    elif fmt == "srt" and out_ext != ".srt":
        print(f"Warning: input is SRT but output filename doesn't end in .srt "
              f"— the file will still be written in SRT format regardless of its name.")
    state = load_progress(args.output_srt)

    if args.redo_range:
        start_id, end_id = args.redo_range
        removed = 0
        for cue_id in range(start_id, end_id + 1):
            if str(cue_id) in state["translations"]:
                del state["translations"][str(cue_id)]
                removed += 1
        print(f"Marked {removed} cue(s) in range {start_id}-{end_id} for re-translation.")

    if args.clean_existing:
        changed = 0
        for k, v in list(state["translations"].items()):
            cleaned = clean_text(v)
            if cleaned != v:
                state["translations"][k] = cleaned
                changed += 1
        save_progress(args.output_srt, {"glossary": state["glossary"], "translations": state["translations"]})
        write_subtitle_file(fmt, raw_data, state["translations"], args.output_srt)
        print(f"Cleaned {changed} already-saved line(s). Updated {args.output_srt}.")
        return

    client = get_client(profile=args.use_key)
    state["client"] = client  # not persisted, just passed around

    done_ids = set(int(k) for k in state["translations"].keys())
    pending = [s for s in subs if s.index not in done_ids]

    if not pending:
        print("Everything already translated according to progress file. Writing final file.")
        write_subtitle_file(fmt, raw_data, {k: v for k, v in state["translations"].items()}, args.output_srt)
        return

    print(f"{len(done_ids)}/{len(subs)} cues already done. {len(pending)} remaining.")

    chunks = [pending[i:i + args.chunk_size] for i in range(0, len(pending), args.chunk_size)]

    for chunk_num, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {chunk_num}/{len(chunks)} (cues {chunk[0].index}-{chunk[-1].index}) ---")

        chunk_items_all = [{"id": s.index, "text": strip_stage_directions(s.content)} for s in chunk]

        # lines that were PURELY a stage direction (now empty) get skipped from
        # the API call entirely and just stored as blank in the output
        chunk_items = []
        for it in chunk_items_all:
            if it["text"]:
                chunk_items.append(it)
            else:
                state["translations"][str(it["id"])] = ""

        if not chunk_items:
            print(f"Chunk {chunk_num}: all lines were stage directions, nothing to translate.")
            save_progress(args.output_srt, {"glossary": state["glossary"], "translations": state["translations"]})
            write_subtitle_file(fmt, raw_data, state["translations"], args.output_srt)
            continue

        try:
            # context: last N already-translated lines, in order
            prior_ids = sorted(int(k) for k in state["translations"].keys())
            context_lines = [state["translations"][str(i)] for i in prior_ids[-CONTEXT_TAIL:]]

            # 1) draft translation (auto-bisects on persistent empty/blocked responses)
            print("  [Phase 1] Translating draft...")
            draft_by_id, permanently_skipped = translate_with_bisection(
                client, chunk_items, state["glossary"], context_lines
            )
            if permanently_skipped:
                # mark with a placeholder so these don't get silently retried forever on
                # every future resume — content-filter blocks are deterministic, not transient
                skipped_originals = {it["id"]: it["text"] for it in chunk_items if it["id"] in permanently_skipped}
                for cue_id in permanently_skipped:
                    state["translations"][str(cue_id)] = "[NEEDS MANUAL TRANSLATION]"
                print(f"  [{len(permanently_skipped)} cue(s) blocked by content filtering, marked for manual translation:]")
                for cue_id in permanently_skipped:
                    print(f"      cue {cue_id}: {skipped_originals.get(cue_id, '')}")

            # only proceed with items that actually got a draft; missing ones stay
            # pending and will be picked up automatically on the next resume run
            chunk_items = [it for it in chunk_items if it["id"] in draft_by_id]
            if not chunk_items:
                save_progress(args.output_srt, {"glossary": state["glossary"], "translations": state["translations"]})
                write_subtitle_file(fmt, raw_data, state["translations"], args.output_srt)
                print(f"Chunk {chunk_num}: nothing successfully translated this round, skipping.")
                continue

            # 2) self-refine pass
            print("  [Phase 2] Reviewing & refining translation...")
            refine_items = [
                {"id": it["id"], "english": it["text"], "draft": draft_by_id.get(it["id"], "")}
                for it in chunk_items
            ]
            refine_prompt = build_refine_prompt(refine_items, state["glossary"])
            raw_refined = call_gemini(client, refine_prompt)
            refined_data = parse_json_array_lenient(raw_refined)
            refined_by_id = {}
            for r in refined_data:
                if "id" in r and "text" in r:
                    refined_by_id[r["id"]] = clean_text(r["text"])

            # any line the refine pass dropped (truncation) just falls back to its draft —
            # better to keep an unrefined-but-valid line than to lose it entirely
            for it in chunk_items:
                if it["id"] not in refined_by_id:
                    refined_by_id[it["id"]] = draft_by_id[it["id"]]

            # 3) flag only lines where refine changed the draft a lot; auto-accept the rest
            print("  [Phase 3] Checking for significant changes / human review...")
            flagged_items = [
                {"id": it["id"], "english": it["text"]}
                for it in chunk_items
                if significant_change(draft_by_id.get(it["id"], ""), refined_by_id.get(it["id"], ""))
            ]

            if flagged_items:
                outcome = review_flagged(flagged_items, refined_by_id, draft_by_id, state)
                if outcome is None:
                    print("Stopped without saving this chunk. Progress up to previous chunk is saved.")
                    save_progress(args.output_srt, {"glossary": state["glossary"], "translations": state["translations"]})
                    write_subtitle_file(fmt, raw_data, state["translations"], args.output_srt)
                    print(f"Partial output written to {args.output_srt}. Re-run the same command to resume.")
                    return
                refined_by_id.update(outcome)
            else:
                print(f"Chunk {chunk_num}: no lines flagged, auto-accepted ({len(chunk_items)} lines).")

            # accept results into state
            for it in chunk_items:
                state["translations"][str(it["id"])] = refined_by_id[it["id"]]

            # 4) update glossary only every 3rd chunk — cuts total API calls by ~1/3
            #    (names/terms don't need re-checking every single chunk to stay consistent)
            if chunk_num % 3 == 0 or chunk_num == len(chunks):
                pairs = [{"english": it["text"], "burmese": refined_by_id[it["id"]]} for it in chunk_items]
                glossary_prompt = build_glossary_prompt(state["glossary"], pairs)
                raw_glossary = call_gemini(client, glossary_prompt)
                try:
                    state["glossary"] = json.loads(re.sub(r"^```(json)?|```$", "", raw_glossary.strip(), flags=re.MULTILINE))
                except Exception:
                    pass  # keep old glossary if this call glitches, not critical

            save_progress(args.output_srt, {"glossary": state["glossary"], "translations": state["translations"]})
            write_subtitle_file(fmt, raw_data, state["translations"], args.output_srt)
            print(f"Chunk {chunk_num} saved. ({len(state['translations'])}/{len(subs)} total cues done)")

        except QuotaExhaustedError as e:
            print(f"\n[STOPPED] {e}")
            save_progress(args.output_srt, {"glossary": state["glossary"], "translations": state["translations"]})
            write_subtitle_file(fmt, raw_data, state["translations"], args.output_srt)
            print(f"Progress saved: {len(state['translations'])}/{len(subs)} cues done in {args.output_srt}.")
            return

    print(f"\nAll done! Final file: {args.output_srt} — open it directly in Aegisub.")


if __name__ == "__main__":
    main()