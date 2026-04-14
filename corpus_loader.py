import os
import re
import json
import shutil
import random
import logging
import tempfile
from pathlib import Path
from typing import List

import nltk

log = logging.getLogger(__name__)

# ── Quality filters ────────────────────────────────────────────────────────────
MIN_LEN     = 5        # minimum token count
MAX_LEN     = 40       # maximum token count
ALPHA_RATIO = 0.60     # fraction of tokens that must be alphabetic
MAX_CORPUS  = 1_000_000  # hard cap to prevent RAM exhaustion

# Kaggle config — API key must be in ~/.kaggle/kaggle.json
KAGGLE_USERNAME = "deepxcodes"

# BookCorpus mirror slugs to try on Kaggle (first that works wins)
BOOKCORPUS_KAGGLE_SLUGS = [
    "enriqueareyan/bookcorpus",
    "ckkissane/bookcorpus-sample",
    "athu1ya/bookcorpus",
]


# ── Quality filter ─────────────────────────────────────────────────────────────
def _quality_sentences(raw_sentences: List[str]) -> List[str]:
    """Filter a list of string sentences to keep only high-quality ones."""
    out = []
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
        # Skip wiki/markdown artefacts
        if sent.startswith(("=", "|", "*", "#", "{", "[")):
            continue
        tokens = sent.split()
        if not (MIN_LEN <= len(tokens) <= MAX_LEN):
            continue
        alpha_count = sum(1 for t in tokens if re.sub(r"[^a-zA-Z]", "", t))
        if alpha_count / len(tokens) < ALPHA_RATIO:
            continue
        out.append(sent.lower())
    return out


# ── Sentence splitter helper ───────────────────────────────────────────────────
def _split_into_sentences(text: str) -> List[str]:
    """Naively split a block of text into individual sentences."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# ══════════════════════════════════════════════════════════════════════════════
#  Local BookCorpus loader
# ══════════════════════════════════════════════════════════════════════════════

def _load_local_bookcorpus(max_sentences: int = 3_000_000) -> List[str]:
    """Load sentences from local bookcorpus_sentences.txt if available."""
    local_path = Path("bookcorpus_sentences.txt")
    if not local_path.exists():
        return []
        
    print(f"  [Local] Loading from '{local_path}' …")
    sents: List[str] = []
    try:
        with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    sents.append(line)
                if len(sents) >= max_sentences:
                    break
        print(f"  [BookCorpus-Local] {len(sents):,} candidate sentences")
        return sents
    except Exception as e:
        print(f"  [Local] failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  HuggingFace loaders
# ══════════════════════════════════════════════════════════════════════════════

def _load_hf_high_quality() -> List[str]:
    """agentlans/high-quality-english-sentences — best curated modern source."""
    try:
        from datasets import load_dataset
        print("  [HF] Downloading agentlans/high-quality-english-sentences …")
        ds = load_dataset("agentlans/high-quality-english-sentences", split="train")
        sents = [row["text"] for row in ds if isinstance(row.get("text"), str)]
        print(f"  [HQ-Sentences]  {len(sents):,} sentences")
        return sents
    except Exception as e:
        print(f"  [HF] high-quality-sentences unavailable: {e}")
        return []


def _load_bookcorpus_hf(max_sentences: int = 3_000_000) -> List[str]:
    """
    BookCorpus via HuggingFace — tries multiple working mirrors in order.

    Why multiple mirrors?
      - 'bookcorpus/bookcorpus' uses a legacy .py dataset script that HuggingFace
        has disabled (raises RuntimeError: Dataset scripts are no longer supported).
      - 'Yuti/bookcorpus'  is a clean Parquet copy — works with streaming=True.
      - 'bookcorpusopen'   is the open subset hosted as Parquet — no scripts needed.
      - 'rojagtap/bookcorpus' is another community Parquet mirror.

    Streaming is used so we never download the full ~5 GB file; we stop once
    *max_sentences* lines have been collected.
    """
    mirrors = [
        # (dataset_id,         config,  text_field, supports_streaming)
        ("Yuti/bookcorpus",    None,    "text",     True),
        ("rojagtap/bookcorpus",None,    "text",     True),
        ("bookcorpusopen",     "plain_text", "text", True),
    ]

    for ds_id, config, field, streaming in mirrors:
        try:
            from datasets import load_dataset
            print(f"  [HF] Trying BookCorpus mirror: {ds_id} …")
            kwargs = dict(split="train", streaming=streaming)
            if config:
                kwargs["name"] = config
            ds = load_dataset(ds_id, **kwargs)

            sents: List[str] = []
            for row in ds:
                text = row.get(field, "")
                if isinstance(text, str) and text.strip():
                    sents.append(text.strip())
                if len(sents) >= max_sentences:
                    break

            if sents:
                print(f"  [BookCorpus-HF] {len(sents):,} sentences  ← {ds_id}")
                return sents

        except Exception as e:
            print(f"  [HF] Mirror {ds_id} failed: {e}")
            continue

    print("  [HF] All BookCorpus HF mirrors failed — will try Kaggle next.")
    return []


def _load_openwebtext(max_sentences: int = 1_500_000) -> List[str]:
    """
    OpenWebText — open recreation of GPT-2's WebText corpus.
    Rich in informal, conversational English; great for intent prediction.
    Streamed to avoid downloading the full ~40 GB dump.
    """
    try:
        from datasets import load_dataset
        print("  [HF] Downloading openwebtext (streaming) …")
        ds = load_dataset("openwebtext", split="train", streaming=True,
                          trust_remote_code=False)
        sents: List[str] = []
        for row in ds:
            text = row.get("text", "")
            if isinstance(text, str) and text.strip():
                sents.extend(_split_into_sentences(text))
            if len(sents) >= max_sentences:
                break
        print(f"  [OpenWebText]   {len(sents):,} candidate sentences")
        return sents[:max_sentences]
    except Exception as e:
        print(f"  [HF] OpenWebText unavailable: {e}")
        return []


def _load_wikitext103() -> List[str]:
    """WikiText-103 — large, clean Wikipedia corpus (~1.8 M training sentences)."""
    try:
        from datasets import load_dataset
        print("  [HF] Downloading wikitext-103 …")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        raw_lines = [row["text"].strip() for row in ds if row.get("text", "").strip()]
        sents = []
        for line in raw_lines:
            if line.startswith("="):   # section headers
                continue
            sents.extend(_split_into_sentences(line))
        print(f"  [WikiText-103]  {len(sents):,} candidate sentences")
        return sents
    except Exception as e:
        print(f"  [HF] WikiText-103 unavailable: {e}")
        return []


def _load_wikitext2() -> List[str]:
    """WikiText-2 — smaller Wikipedia fallback."""
    try:
        from datasets import load_dataset
        print("  [HF] Downloading wikitext-2 …")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        raw_lines = [row["text"].strip() for row in ds if row.get("text", "").strip()]
        sents = []
        for line in raw_lines:
            if line.startswith("="):
                continue
            sents.extend(_split_into_sentences(line))
        print(f"  [WikiText-2]    {len(sents):,} candidate sentences")
        return sents
    except Exception as e:
        print(f"  [HF] WikiText-2 unavailable: {e}")
        return []


def _load_lambada() -> List[str]:
    """
    LAMBADA — book passages designed for sentence completion.
    Each record is a paragraph; we split it into constituent sentences.
    """
    try:
        from datasets import load_dataset
        print("  [HF] Downloading LAMBADA …")
        ds = load_dataset("lambada", split="train", trust_remote_code=True)
        sents = []
        for row in ds:
            text = row.get("text", "")
            if isinstance(text, str) and text.strip():
                sents.extend(_split_into_sentences(text))
        print(f"  [LAMBADA]       {len(sents):,} candidate sentences")
        return sents
    except Exception as e:
        print(f"  [HF] LAMBADA unavailable: {e}")
        return []


def _load_hellaswag() -> List[str]:
    """
    HellaSwag — grounded sentence completion dataset.
    ctx  = sentence context (beginning)
    endings[label] = correct continuation

    We concatenate ctx + correct ending to form a complete, natural sentence.
    """
    try:
        from datasets import load_dataset
        print("  [HF] Downloading HellaSwag …")
        ds = load_dataset("Rowan/hellaswag", split="train")
        sents = []
        for row in ds:
            ctx     = row.get("ctx", "").strip()
            endings = row.get("endings", [])
            label   = row.get("label", "")
            try:
                idx = int(label)
                if ctx and 0 <= idx < len(endings):
                    full = (ctx + " " + endings[idx]).strip()
                    sents.append(full)
                    # Also add the context alone if long enough
                    if len(ctx.split()) >= MIN_LEN:
                        sents.append(ctx)
            except (ValueError, TypeError):
                pass
        print(f"  [HellaSwag]     {len(sents):,} candidate sentences")
        return sents
    except Exception as e:
        print(f"  [HF] HellaSwag unavailable: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  Kaggle loader — BookCorpus mirror (fallback when HF mirrors fail)
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_kaggle_auth() -> bool:
    """
    Check that kaggle.json exists at the standard location and contains the
    correct username.  Returns True if auth is likely to work.

    Location (Windows): C:/Users/<user>/.kaggle/kaggle.json
    Location (Linux/Mac): ~/.kaggle/kaggle.json
    """
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(
            f"  [Kaggle] ⚠  kaggle.json not found at {kaggle_json}\n"
            "           Download it from https://kaggle.com → Account → API → Create New Token\n"
            "           and place it at that path to enable BookCorpus download."
        )
        return False

    try:
        creds = json.loads(kaggle_json.read_text())
        if creds.get("username") != KAGGLE_USERNAME:
            print(
                f"  [Kaggle] ⚠  kaggle.json username is '{creds.get('username')}' "
                f"but expected '{KAGGLE_USERNAME}'. Update kaggle.json."
            )
            return False
        # Ensure file permissions are restricted (Kaggle library requirement)
        try:
            kaggle_json.chmod(0o600)
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"  [Kaggle] ⚠  Could not parse kaggle.json: {e}")
        return False


def _load_bookcorpus_kaggle(max_sentences: int = 3_000_000) -> List[str]:
    """
    Download a BookCorpus mirror from Kaggle using the Kaggle Python API.
    Tries several known community-uploaded dataset slugs in order.
    Returns up to *max_sentences* raw sentences.
    Only called when ALL HuggingFace BookCorpus mirrors have failed.
    """
    if not _ensure_kaggle_auth():
        return []

    try:
        import kaggle  # noqa — imported to trigger auth from kaggle.json
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
    except ImportError:
        print("  [Kaggle] kaggle package not installed. Run: pip install kaggle")
        return []

    api = KaggleApiExtended()
    try:
        api.authenticate()
    except Exception as e:
        print(f"  [Kaggle] Authentication failed: {e}")
        return []

    tmp_dir = Path(tempfile.mkdtemp(prefix="synapse_bookcorpus_"))
    sents: List[str] = []

    for slug in BOOKCORPUS_KAGGLE_SLUGS:
        try:
            print(f"  [Kaggle] Trying BookCorpus slug: {slug} …")
            api.dataset_download_files(slug, path=str(tmp_dir), unzip=True, quiet=False)

            # Walk all text/CSV files in the download directory
            for fpath in tmp_dir.rglob("*"):
                if fpath.suffix.lower() not in {".txt", ".csv", ".tsv"}:
                    continue
                try:
                    text = fpath.read_text(encoding="utf-8", errors="ignore")
                    for line in text.splitlines():
                        line = line.strip()
                        if line:
                            sents.extend(_split_into_sentences(line))
                        if len(sents) >= max_sentences:
                            break
                except Exception:
                    continue
                if len(sents) >= max_sentences:
                    break

            if sents:
                print(f"  [BookCorpus-Kaggle] {len(sents):,} raw sentences from {slug}")
                break   # found a working slug — stop trying

        except Exception as e:
            print(f"  [Kaggle] Slug {slug} failed: {e}")
            continue

    # Clean up temp download directory
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    return sents[:max_sentences]


# ══════════════════════════════════════════════════════════════════════════════
#  NLTK loaders (offline, always available)
# ══════════════════════════════════════════════════════════════════════════════

def _load_nltk_corpora() -> List[str]:
    """Load and merge all available NLTK sentence corpora."""
    sentences = []

    corpora = [
        ("brown",     "brown",     "Brown"),
        ("gutenberg", "gutenberg", "Gutenberg"),
        ("reuters",   "reuters",   "Reuters"),
        ("inaugural", "inaugural", "Inaugural"),
        ("webtext",   "webtext",   "Webtext"),
    ]

    for dl_id, attr, name in corpora:
        try:
            nltk.download(dl_id, quiet=True)
            import importlib
            corp = importlib.import_module("nltk.corpus").__dict__[attr]
            sents = [" ".join(s) for s in corp.sents()]
            sentences.extend(sents)
            print(f"  [NLTK {name:<10}] {len(sents):,} sentences")
        except Exception as e:
            log.warning(f"NLTK {name} unavailable: {e}")

    return sentences


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def load_best_corpus(max_sentences: int = MAX_CORPUS) -> List[str]:
    print("\n" + "═" * 60)
    print("  Synapse Corpus Loader — collecting training sentences")
    print("═" * 60)

    raw: List[str] = []

    # ── Priority 0: Local Bookcorpus txt ───────────────────────────────────────
    local = _load_local_bookcorpus(max_sentences=max_sentences)
    raw.extend(local)
    print(f"  → Running total: {len(raw):,}")

    # ── Priority 1: Best curated modern HuggingFace sentences ─────────────────
    if len(raw) < max_sentences:
        hq = _load_hf_high_quality()
        raw.extend(hq)
        print(f"  → Running total: {len(raw):,}")

    # ── Priority 2: BookCorpus (HF Parquet mirrors, no legacy scripts) ────────
    if len(raw) < max_sentences:
        bc_hf = _load_bookcorpus_hf(max_sentences=3_000_000)
        raw.extend(bc_hf)
        print(f"  → Running total: {len(raw):,}")

    # ── Priority 3: WikiText-103 (large Wikipedia) ────────────────────────────
    if len(raw) < max_sentences:
        wt103 = _load_wikitext103()
        raw.extend(wt103)
        print(f"  → Running total: {len(raw):,}")

    # ── Priority 4: LAMBADA (book passage completions) ────────────────────────
    if len(raw) < max_sentences:
        lb = _load_lambada()
        raw.extend(lb)
        print(f"  → Running total: {len(raw):,}")

    # ── Priority 5: HellaSwag (grounded sentence completions) ─────────────────
    if len(raw) < max_sentences:
        hs = _load_hellaswag()
        raw.extend(hs)
        print(f"  → Running total: {len(raw):,}")

    # Priority 6: OpenWebText (conversational web text)
    if len(raw) < max_sentences:
        owt = _load_openwebtext(max_sentences=1_500_000)
        raw.extend(owt)
        print(f"  → Running total: {len(raw):,}")

    # Priority 7: Kaggle BookCorpus mirror 
    #    Only attempted when HF BookCorpus mirrors all failed AND we still need
    #    more data.
    if len(raw) < max_sentences:
        bc_kaggle = _load_bookcorpus_kaggle(max_sentences=3_000_000)
        raw.extend(bc_kaggle)
        print(f"  → Running total: {len(raw):,}")

    # Priority 8: WikiText-2 fallback
    if len(raw) < 800_000:
        wt2 = _load_wikitext2()
        raw.extend(wt2)
        print(f"  → Running total: {len(raw):,}")

    # Priority 9: NLTK corpora 
    nltk_sents = _load_nltk_corpora()
    raw.extend(nltk_sents)
    print(f"  → Running total: {len(raw):,}")

    # Quality filter
    print(f"\n  Raw collected:    {len(raw):,} sentences")
    filtered = _quality_sentences(raw)
    print(f"  After filtering:  {len(filtered):,} sentences")

    # Deduplicate
    seen: set = set()
    unique: List[str] = []
    for s in filtered:
        key = s.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(s)
    print(f"  After dedup:      {len(unique):,} sentences")

    # ── Shuffle and cap ───────────────────────────────────────────────────────
    random.shuffle(unique)
    final = unique[:max_sentences]
    print(f"  Final corpus:     {len(final):,} sentences  (cap={max_sentences:,})")
    print("═" * 60 + "\n")
    return final