import random
import re
import os
import pickle
import numpy as np
from functools import lru_cache
from nltk.tokenize import word_tokenize
from nltk import pos_tag, pos_tag_sents
from collections import defaultdict, Counter

# ─────────────────────────────────────────────
# Sentinels & frozen sets
# ─────────────────────────────────────────────
__START__ = "__START__"
__END__   = "__END__"
_UNK_     = "<UNK>"

_EMIT_BLACKLIST = frozenset({_UNK_, __START__, __END__, "<START>", "<END>"})

_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "of", "in", "to", "is", "was", "and", "or",
    "for", "on", "at", "by", "it", "that", "this", "with", "as",
    "be", "are", "were", "been", "not", "but", "from", "has", "had",
    "have", "do", "did", "will", "would", "could", "should", "may",
    "might", "can", "shall", "he", "she", "they", "we", "i", "you",
    "its", "his", "her", "their", "our", "my", "your",
})

_TRAILING_FORBIDDEN = frozenset({
    "of", "in", "the", "a", "an", "for", "to", "by",
    "on", "at", "and", "or", "but", "with", "as",
})

# Default path for persisted model
_DEFAULT_MODEL_PATH = "hmm_model.pkl"


# ─────────────────────────────────────────────
# Temperature sampler
# ─────────────────────────────────────────────
def temperature_sample(options: list, probabilities: list, temp: float):
    """Sample from *options* weighted by *probabilities* scaled by temperature."""
    if not options:
        return None
    if temp <= 0.01:
        best = max(zip(options, probabilities), key=lambda x: x[1])
        return best[0]
    probs = np.array(probabilities, dtype=np.float64)
    scaled = np.where(probs > 0, probs ** (1.0 / temp), 0.0)
    total = scaled.sum()
    if total == 0.0:
        return random.choice(options)
    norm = (scaled / total).tolist()
    return random.choices(options, weights=norm)[0]


# ─────────────────────────────────────────────
# Cached tokeniser
# ─────────────────────────────────────────────
@lru_cache(maxsize=4096)
def _cached_tokenize(text: str) -> tuple:
    return tuple(word_tokenize(text.lower()))


# ─────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────
def _clean_output(raw: str) -> str:
    """Produce a clean, human-readable sentence from raw token string."""
    words = [w for w in raw.split() if w != "..."]
    if not words:
        return raw

    # Capitalize standalone pronoun 'i'
    words = ["I" if w == "i" else w for w in words]

    # Collapse consecutive duplicate content words
    deduped = [words[0]]
    for w in words[1:]:
        if w.lower() != deduped[-1].lower() or w.lower() in _FUNCTION_WORDS:
            deduped.append(w)
    words = deduped

    # Strip trailing dangling function words
    while words and words[-1].lower() in _TRAILING_FORBIDDEN:
        words.pop()
    if not words:
        return raw

    sentence = " ".join(words)
    sentence = sentence[0].upper() + sentence[1:]
    sentence = re.sub(r" {2,}", " ", sentence)
    if sentence[-1] not in ".!?":
        sentence += "."
    return sentence


# ─────────────────────────────────────────────
# HMM model
# ─────────────────────────────────────────────
class HMMGenerator:

    def __init__(self):
        # Raw counts
        self.transition_counts:   defaultdict = defaultdict(Counter)
        self.transition_counts_2: defaultdict = defaultdict(Counter)   # trigram
        self.emission_counts:     defaultdict = defaultdict(Counter)
        self.tag_pair_emission:   defaultdict = defaultdict(Counter)   # P4
        self.word_bigram_counts:  defaultdict = defaultdict(Counter)
        self.start_counts:        Counter     = Counter()
        self.vocabulary: set = set()
        self.tags:       set = set()

        # Sentence length statistics (P2)
        self._sent_len_total: int = 0
        self._sent_len_count: int = 0

        # POS cache
        self._pos_cache: dict = {}

        # Probability tables (built by build_probs)
        self.emission_probs:     dict = {}
        self.transition_probs:   dict = {}
        self.transition_probs_2: dict = {}   # trigram
        self.tag_pair_emit_probs:dict = {}   # P4
        self.word_bigram_probs:  dict = {}
        self.start_probs:        dict = {}
        self.avg_sentence_length: float = 10.0

        # Fast lookup caches
        self._top_emissions:    dict = {}   # tag -> [(word, prob), …]
        self._clean_transitions:dict = {}   # tag -> [(next_tag, prob), …]

    # ── Helpers ──────────────────────────────
    def _tag(self, text: str) -> list:
        if text not in self._pos_cache:
            tokens = list(_cached_tokenize(text))
            self._pos_cache[text] = pos_tag(tokens)
        return self._pos_cache[text]

    # ── Core training logic ──────────────────
    def train(self, corpus, user_sentences=None):
        """
        Train on an iterable of sentence strings.
        user_sentences (optional list) are weighted 5× for domain boosting.
        """
        if user_sentences is None:
            user_sentences = []

        # Reset everything
        self.transition_counts.clear()
        self.transition_counts_2.clear()
        self.emission_counts.clear()
        self.tag_pair_emission.clear()
        self.word_bigram_counts.clear()
        self.start_counts.clear()
        self.vocabulary.clear()
        self.tags.clear()
        self._pos_cache.clear()
        self._sent_len_total = 0
        self._sent_len_count = 0
        _cached_tokenize.cache_clear()

        self._process(list(corpus), weight=1)
        if user_sentences:
            self._process(list(user_sentences), weight=5)

    def train_from_file(
        self,
        file_path: str,
        max_sentences: int = 300_000,
        batch_size: int = 5_000,
        user_sentences=None,
        verbose: bool = True,
    ):
        """
        Train directly from a plain-text corpus file (one sentence per line).
        Designed for large files like BookCorpus.

        Parameters
        ----------
        file_path      : path to .txt corpus (one sentence per line)
        max_sentences  : cap on how many lines to read (default 300 000)
        batch_size     : lines processed per progress report
        user_sentences : optional list of domain sentences weighted 5×
        verbose        : print progress to stdout
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Corpus file not found: '{file_path}'\n"
                "Run download_bookcorpus.py first to generate it."
            )

        # Reset state before incremental loading
        self.transition_counts.clear()
        self.transition_counts_2.clear()
        self.emission_counts.clear()
        self.tag_pair_emission.clear()
        self.word_bigram_counts.clear()
        self.start_counts.clear()
        self.vocabulary.clear()
        self.tags.clear()
        self._pos_cache.clear()
        self._sent_len_total = 0
        self._sent_len_count = 0
        _cached_tokenize.cache_clear()

        if verbose:
            print(f"[HMMGenerator] Reading corpus: {file_path}")
            print(f"[HMMGenerator] Max sentences : {max_sentences:,}")

        buffer = []
        total_loaded = 0

        with open(file_path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                if total_loaded >= max_sentences:
                    break
                line = raw_line.strip()
                if not line:
                    continue
                buffer.append(line)
                total_loaded += 1

                if len(buffer) >= batch_size:
                    self._process(buffer, weight=1)
                    buffer.clear()
                    if verbose:
                        print(f"  … processed {total_loaded:,} sentences", end="\r")

        # Flush remaining
        if buffer:
            self._process(buffer, weight=1)

        if verbose:
            print(f"\n[HMMGenerator] Corpus loaded  : {total_loaded:,} sentences")
            print(f"[HMMGenerator] Vocabulary size: {len(self.vocabulary):,}")

        # Boost user / domain sentences
        if user_sentences:
            if verbose:
                print(f"[HMMGenerator] Boosting {len(user_sentences):,} user sentences (weight=5)")
            self._process(list(user_sentences), weight=5)

    def _process(self, sentences: list, weight: int):
        """Accumulate raw counts from a list of sentence strings."""
        if not sentences:
            return

        # Batch POS-tag uncached sentences
        uncached_keys, uncached_tokens = [], []
        for sent in sentences:
            toks = list(_cached_tokenize(sent))
            if sent not in self._pos_cache:
                uncached_keys.append(sent)
                uncached_tokens.append(toks)

        if uncached_tokens:
            batch_tagged = pos_tag_sents(uncached_tokens)
            for key, tagged in zip(uncached_keys, batch_tagged):
                self._pos_cache[key] = tagged

        # Accumulate counts
        for sent in sentences:
            cached = self._pos_cache.get(sent, [])
            filtered = [(w, t) for w, t in cached if w.isalpha()]
            if not filtered:
                continue

            real_words = [w for w, _ in filtered]
            real_tags  = [t for _, t in filtered]

            # Sentence length (P2)
            self._sent_len_total += len(real_words)
            self._sent_len_count += 1

            words = [__START__] + real_words + [__END__]
            tags  = [__START__] + real_tags  + [__END__]

            prev_prev_tag = None
            prev_tag      = None
            prev_word     = None

            for i, (word, tag) in enumerate(zip(words, tags)):
                self.vocabulary.add(word)
                self.tags.add(tag)

                if i == 1:
                    self.start_counts[tag] += weight

                if word not in (_EMIT_BLACKLIST | {__START__, __END__}):
                    self.emission_counts[tag][word] += weight
                    if prev_tag is not None and prev_tag not in {__START__}:
                        self.tag_pair_emission[(prev_tag, tag)][word] += weight

                if prev_tag is not None:
                    self.transition_counts[prev_tag][tag] += weight

                if prev_prev_tag is not None and prev_tag is not None:
                    self.transition_counts_2[(prev_prev_tag, prev_tag)][tag] += weight

                if prev_word is not None and word not in {__START__, __END__}:
                    self.word_bigram_counts[prev_word][word] += weight

                prev_prev_tag = prev_tag
                prev_tag      = tag
                prev_word     = word if word not in {__START__, __END__} else prev_word

    # ── Probability tables ────────────────────
    def _smooth(self, counts_dict: dict, vocab_size: int, k: float) -> dict:
        probs = {}
        for key, counter in counts_dict.items():
            total = sum(counter.values())
            denom = total + k * vocab_size
            probs[key] = {w: (v + k) / denom for w, v in counter.items()}
            probs[key][_UNK_] = k / denom if k > 0 else 0.0
        return probs

    def build_probs(self, k: float = 0.001):
        """Compute all smoothed probability tables and fast-lookup caches."""
        V = len(self.vocabulary)
        T = len(self.tags)

        self.emission_probs      = self._smooth(self.emission_counts, V, k)
        self.tag_pair_emit_probs = self._smooth(self.tag_pair_emission, V, k)

        # Bigram transition
        self.transition_probs = {}
        for key, counter in self.transition_counts.items():
            total = sum(counter.values())
            denom = total + k * T
            self.transition_probs[key] = {
                tag: (v + k) / denom for tag, v in counter.items()
            }
            self.transition_probs[key][_UNK_] = k / denom if k > 0 else 0.0

        # Trigram transition
        self.transition_probs_2 = {}
        for key, counter in self.transition_counts_2.items():
            total = sum(counter.values())
            denom = total + k * T
            self.transition_probs_2[key] = {
                tag: (v + k) / denom for tag, v in counter.items()
            }
            self.transition_probs_2[key][_UNK_] = k / denom if k > 0 else 0.0

        # Start probs
        start_total = sum(self.start_counts.values())
        self.start_probs = {
            tag: count / start_total if start_total > 0 else 0.0
            for tag, count in self.start_counts.items()
        }

        # Word bigram probs
        self.word_bigram_probs = {}
        for prev_word, counter in self.word_bigram_counts.items():
            total = sum(counter.values())
            denom = total + k * V
            self.word_bigram_probs[prev_word] = {
                w: (v + k) / denom for w, v in counter.items()
            }

        # Average sentence length (P2)
        self.avg_sentence_length = (
            self._sent_len_total / self._sent_len_count
            if self._sent_len_count > 0 else 10.0
        )

        # Top-50 emissions cache
        self._top_emissions = {}
        for tag, word_probs in self.emission_probs.items():
            candidates = [
                (w, p) for w, p in word_probs.items()
                if w not in _EMIT_BLACKLIST and w not in {__START__, __END__}
            ]
            candidates.sort(key=lambda x: x[1], reverse=True)
            self._top_emissions[tag] = candidates[:50]

        # Clean transitions cache
        self._clean_transitions = {}
        _exclude = frozenset({_UNK_, __START__})
        for tag, tag_probs in self.transition_probs.items():
            self._clean_transitions[tag] = [
                (t, p) for t, p in tag_probs.items()
                if t not in _exclude
            ]

    # ── Save / Load ───────────────────────────
    def save(self, path: str = _DEFAULT_MODEL_PATH):
        """
        Persist the trained model to disk using pickle.
        Saves both raw counts and pre-built probability tables.
        """
        payload = {
            # Raw counts (needed to resume incremental training)
            "transition_counts":   dict(self.transition_counts),
            "transition_counts_2": dict(self.transition_counts_2),
            "emission_counts":     dict(self.emission_counts),
            "tag_pair_emission":   dict(self.tag_pair_emission),
            "word_bigram_counts":  dict(self.word_bigram_counts),
            "start_counts":        dict(self.start_counts),
            "vocabulary":          self.vocabulary,
            "tags":                self.tags,
            "_sent_len_total":     self._sent_len_total,
            "_sent_len_count":     self._sent_len_count,
            # Pre-built probability tables
            "emission_probs":      self.emission_probs,
            "transition_probs":    self.transition_probs,
            "transition_probs_2":  self.transition_probs_2,
            "tag_pair_emit_probs": self.tag_pair_emit_probs,
            "word_bigram_probs":   self.word_bigram_probs,
            "start_probs":         self.start_probs,
            "avg_sentence_length": self.avg_sentence_length,
            "_top_emissions":      self._top_emissions,
            "_clean_transitions":  self._clean_transitions,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"[HMMGenerator] Model saved  → {path}  ({size_mb:.1f} MB)")

    def load(self, path: str = _DEFAULT_MODEL_PATH):
        """
        Load a previously saved model from disk.
        Raises FileNotFoundError if *path* does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[HMMGenerator] No saved model found at '{path}'.\n"
                "Run train.py first to train and save the model."
            )
        with open(path, "rb") as fh:
            data = pickle.load(fh)

        # Restore raw counts
        self.transition_counts   = defaultdict(Counter, {k: Counter(v) for k, v in data["transition_counts"].items()})
        self.transition_counts_2 = defaultdict(Counter, {k: Counter(v) for k, v in data["transition_counts_2"].items()})
        self.emission_counts     = defaultdict(Counter, {k: Counter(v) for k, v in data["emission_counts"].items()})
        self.tag_pair_emission   = defaultdict(Counter, {k: Counter(v) for k, v in data["tag_pair_emission"].items()})
        self.word_bigram_counts  = defaultdict(Counter, {k: Counter(v) for k, v in data["word_bigram_counts"].items()})
        self.start_counts        = Counter(data["start_counts"])
        self.vocabulary          = data["vocabulary"]
        self.tags                = data["tags"]
        self._sent_len_total     = data["_sent_len_total"]
        self._sent_len_count     = data["_sent_len_count"]

        # Restore probability tables
        self.emission_probs      = data["emission_probs"]
        self.transition_probs    = data["transition_probs"]
        self.transition_probs_2  = data["transition_probs_2"]
        self.tag_pair_emit_probs = data["tag_pair_emit_probs"]
        self.word_bigram_probs   = data["word_bigram_probs"]
        self.start_probs         = data["start_probs"]
        self.avg_sentence_length = data["avg_sentence_length"]
        self._top_emissions      = data["_top_emissions"]
        self._clean_transitions  = data["_clean_transitions"]

        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(
            f"[HMMGenerator] Model loaded ← {path}  ({size_mb:.1f} MB)\n"
            f"               Vocabulary : {len(self.vocabulary):,} words\n"
            f"               Avg length : {self.avg_sentence_length:.1f} tokens"
        )

    # ── Predict: top-N next words ─────────────
    def predict(self, partial_sentence: str, top_n: int = 5) -> list:
        """
        Given an incomplete sentence, return up to *top_n* candidate next words
        ranked by a blend of word-bigram and tag-pair emission probabilities.

        Parameters
        ----------
        partial_sentence : incomplete input text (e.g. "she walked into the")
        top_n            : number of suggestions to return

        Returns
        -------
        list of (word, score) tuples, best first.
        """
        if not self._top_emissions:
            raise RuntimeError("Model not built – call build_probs() after training.")

        words = partial_sentence.strip().lower().split()
        if not words:
            return []

        last_word = words[-1]

        # ── 1. Word-bigram candidates ─────────
        bigram_scores: dict = {}
        if last_word in self.word_bigram_probs:
            bigram_scores = self.word_bigram_probs[last_word]

        # ── 2. Tag-based candidates ───────────
        tagged = self._tag(partial_sentence)
        if not tagged:
            return []

        current_tag = tagged[-1][1]
        prev_tag    = tagged[-2][1] if len(tagged) >= 2 else None

        # Decide which transition distribution to use
        tri_key = (prev_tag, current_tag) if prev_tag else None
        if tri_key and tri_key in self.transition_probs_2:
            trans_dist = self.transition_probs_2[tri_key]
        else:
            trans_dist = self.transition_probs.get(current_tag, {})

        # Collect top emission words for the most probable next tags
        tag_word_scores: Counter = Counter()
        for next_tag, tag_prob in sorted(trans_dist.items(), key=lambda x: -x[1])[:10]:
            if next_tag in {__END__, _UNK_, __START__}:
                continue
            pair_key = (current_tag, next_tag)
            if pair_key in self.tag_pair_emit_probs:
                emit_dist = self.tag_pair_emit_probs[pair_key]
            else:
                emit_dist = self.emission_probs.get(next_tag, {})

            for word, emit_prob in emit_dist.items():
                if word in _EMIT_BLACKLIST or not word.isalpha():
                    continue
                tag_word_scores[word] += tag_prob * emit_prob

        # ── 3. Blend bigram + tag scores ──────
        all_words = set(tag_word_scores) | set(bigram_scores)
        blended: dict = {}
        for w in all_words:
            if w in _EMIT_BLACKLIST or not w.isalpha():
                continue
            ts = tag_word_scores.get(w, 0.0)
            bs = bigram_scores.get(w, 0.0)
            blended[w] = 0.55 * ts + 0.45 * bs   # favour tag signal slightly

        ranked = sorted(blended.items(), key=lambda x: -x[1])
        return ranked[:top_n]

    # ── Complete: return full sentence string ─
    def complete_sentence(
        self,
        partial_sentence: str,
        max_words:  int   = 15,
        temperature: float = 0.55,
    ) -> str:
        """
        Convenience wrapper around autocomplete().
        Returns only the completed sentence string (no reasoning log).

        Parameters
        ----------
        partial_sentence : incomplete input (e.g. "the sun was setting over")
        max_words        : maximum new words to generate
        temperature      : sampling temperature (lower = more deterministic)

        Returns
        -------
        str – a clean, punctuated sentence.
        """
        completed, _ = self.autocomplete(
            prompt=partial_sentence,
            max_length=max_words,
            temperature=temperature,
        )
        return completed

    # ── Inference (original, fully preserved) ─
    def autocomplete(
        self,
        prompt:      str,
        max_length:  int   = 15,
        temperature: float = 0.55,
    ) -> tuple:
        """
        Generate a sentence completion for *prompt*.

        Returns
        -------
        (completed_sentence: str, reasoning_log: list)
        """
        if not self._top_emissions:
            raise RuntimeError("Model not trained – call train() then build_probs().")

        tagged = self._tag(prompt)
        if not tagged:
            return prompt, []

        input_words  = [w for w, _ in tagged]
        current_tag  = tagged[-1][1]
        prev_tag     = tagged[-2][1] if len(tagged) >= 2 else None

        sentence_completion: list = []
        reasoning_log:       list = []
        recent_words:        list = []

        for step in range(max_length):
            # Transition: trigram → bigram fallback
            tri_key = (prev_tag, current_tag) if prev_tag else None
            if tri_key and tri_key in self.transition_probs_2:
                tri_dist = self.transition_probs_2[tri_key]
                trans_options = [
                    (t, p) for t, p in tri_dist.items()
                    if t not in {_UNK_, __START__}
                ]
            else:
                trans_options = self._clean_transitions.get(current_tag)

            if not trans_options:
                if not self.start_probs:
                    break
                current_tag = temperature_sample(
                    list(self.start_probs.keys()),
                    list(self.start_probs.values()),
                    temp=temperature,
                )
                prev_tag = None
                reasoning_log.append(f"Dead-end -> jumped to: {current_tag}")
                continue

            next_tags = [t for t, _ in trans_options]
            probs     = list(np.array([p for _, p in trans_options], dtype=np.float64))

            # P2: sentence-length END boost
            gen_len = len(sentence_completion)
            if gen_len >= self.avg_sentence_length:
                excess    = gen_len - self.avg_sentence_length + 1
                end_boost = float(np.exp(0.7 * excess))
                probs = [
                    p * end_boost if t == __END__ else p
                    for t, p in zip(next_tags, probs)
                ]

            next_tag = temperature_sample(next_tags, probs, temp=temperature)

            if next_tag == __END__:
                reasoning_log.append(f"{current_tag} -> __END__ | natural stop")
                break

            # Word generation: 3-way contextual blend
            top = self._top_emissions.get(next_tag)
            if not top:
                word = "..."
            else:
                word_choices = [w for w, _ in top]
                ep_scores    = np.array([p for _, p in top], dtype=np.float64)

                # (a) Tag-pair emission
                pair_key = (current_tag, next_tag)
                if pair_key in self.tag_pair_emit_probs:
                    te = self.tag_pair_emit_probs[pair_key]
                    te_scores = np.array(
                        [te.get(w, te.get(_UNK_, 0.0)) for w in word_choices],
                        dtype=np.float64,
                    )
                else:
                    te_scores = ep_scores.copy()

                # (b) Word bigram
                last_word = (
                    recent_words[-1] if recent_words
                    else (input_words[-1] if input_words else None)
                )
                if last_word and last_word in self.word_bigram_probs:
                    bi = self.word_bigram_probs[last_word]
                    bi_scores = np.array(
                        [bi.get(w, 0.0) for w in word_choices],
                        dtype=np.float64,
                    )
                else:
                    bi_scores = ep_scores.copy()

                # (c) 3-way blend: 40% emission + 35% tag-pair + 25% word bigram
                blended = 0.40 * ep_scores + 0.35 * te_scores + 0.25 * bi_scores

                # (d) Repetition penalty on content words in last 6 tokens
                recent_set = Counter(recent_words[-6:])
                for idx, w in enumerate(word_choices):
                    if w in recent_set and w not in _FUNCTION_WORDS:
                        blended[idx] *= (0.05 ** recent_set[w])

                word = temperature_sample(word_choices, blended.tolist(), temp=temperature)

            reasoning_log.append(f"{current_tag} -> {next_tag} | '{word}'")
            sentence_completion.append(word)
            recent_words.append(word)

            prev_tag    = current_tag
            current_tag = next_tag

        raw = " ".join(input_words + sentence_completion)
        return _clean_output(raw), reasoning_log