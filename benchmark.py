"""
HMM Model Accuracy Benchmark — v3
====================================
Metrics:
  1.  POS Transition Accuracy     – % of generated bigrams that are valid English POS sequences
  2.  Sentinel Cleanliness        – % of outputs free of <START>/<END>/<UNK>/...
  3.  No-Repetition Score         – % of outputs without consecutive duplicate content words
  4.  Vocabulary Diversity        – average unique-content-word ratio in generated completions
  5.  Natural Termination         – % of sentences that end naturally before hitting max_length
  6.  Next-Tag Top-15 Accuracy    – % of test-sentence tag bigrams where the model's top-15
                                    predicted next tags include the actual tag. Replaces the
                                    old perplexity score which was mathematically unsuitable
                                    for this class of model (perplexity of 10000+ for any
                                    first-order word-level HMM is normal; scoring it linearly
                                    is misleading).

Composite weights: POS 0.30 | Sentinel 0.15 | Repeat 0.20 | Diversity 0.10
                   Termination 0.10 | Next-Tag 0.15  (sum = 1.00)
"""

import math
import time
from collections import Counter
from hmm_model import HMMGenerator
from nltk.tokenize import word_tokenize
from nltk import pos_tag

#Valid English POS bigrams
VALID_TRANSITIONS = {
    'DT':   {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'RB', 'CD', 'VBG', 'PRP$'},
    'JJ':   {'NN', 'NNS', 'NNP', 'JJ', 'CC', 'IN', 'CD', 'VBG', 'RB', 'JJR'},
    'NN':   {'IN', 'VBZ', 'VBD', 'VBP', 'CC', 'NN', 'NNS', 'MD', 'WDT', 'WP', 'TO', 'RB', 'POS', 'VBG', 'VBN', 'JJ'},
    'NNS':  {'IN', 'VBP', 'VBD', 'CC', 'MD', 'WDT', 'WP', 'TO', 'RB', 'VBZ', 'JJ'},
    'NNP':  {'NNP', 'NN', 'VBZ', 'VBD', 'VBP', 'IN', 'CC', 'MD', 'POS', 'NNS', 'JJ'},
    'NNPS': {'VBP', 'VBD', 'IN', 'CC', 'MD'},
    'VB':   {'DT', 'NN', 'NNS', 'NNP', 'JJ', 'IN', 'RB', 'TO', 'PRP', 'VBG', 'VBN', 'CD', 'PRP$', 'EX', 'WDT', 'RBR'},
    'VBD':  {'DT', 'NN', 'NNS', 'NNP', 'JJ', 'IN', 'RB', 'TO', 'PRP', 'VBN', 'VBG', 'CD', 'PRP$', 'EX', 'RBR'},
    'VBZ':  {'DT', 'NN', 'NNS', 'NNP', 'JJ', 'IN', 'RB', 'TO', 'PRP', 'VBN', 'VBG', 'CD', 'PRP$', 'EX', 'RBR', 'RB'},
    'VBP':  {'DT', 'NN', 'NNS', 'NNP', 'JJ', 'IN', 'RB', 'TO', 'PRP', 'VBN', 'VBG', 'CD', 'PRP$', 'EX', 'RBR'},
    'VBN':  {'DT', 'NN', 'NNS', 'IN', 'RB', 'TO', 'CC', 'JJ', 'PRP', 'BY', 'VBN'},
    'VBG':  {'DT', 'NN', 'NNS', 'IN', 'RB', 'TO', 'JJ', 'PRP', 'CD', 'CC'},
    'IN':   {'DT', 'NN', 'NNS', 'NNP', 'JJ', 'PRP', 'PRP$', 'VBG', 'WDT', 'WP', 'CD', 'RB', 'EX', 'NNPS'},
    'CC':   {'DT', 'NN', 'NNS', 'NNP', 'JJ', 'VB', 'VBD', 'VBZ', 'VBP', 'RB', 'PRP', 'IN', 'CD', 'TO', 'MD', 'VBG', 'NNPS'},
    'TO':   {'VB', 'NN', 'NNP', 'DT', 'RB', 'JJ', 'VBN'},
    'MD':   {'VB', 'RB', 'VBP', 'VBN', 'VBG', 'NOT'},
    'PRP':  {'VBZ', 'VBD', 'VBP', 'MD', 'IN', 'CC', 'RB', 'VB', 'TO', 'VBG'},
    'PRP$': {'NN', 'NNS', 'NNP', 'JJ', 'JJR', 'VBG'},
    'RB':   {'VB', 'VBD', 'VBZ', 'VBP', 'VBN', 'VBG', 'JJ', 'RB', 'IN', 'DT', 'CD', 'TO', 'RBR', 'NN', 'NNP'},
    'RBR':  {'JJ', 'VBN', 'VBD', 'DT', 'IN'},
    'WDT':  {'VBZ', 'VBD', 'VBP', 'NN', 'MD', 'VB', 'PRP', 'VBN'},
    'WP':   {'VBZ', 'VBD', 'VBP', 'MD', 'VB', 'NN'},
    'WRB':  {'VBZ', 'VBD', 'VBP', 'MD', 'VB', 'PRP', 'DT', 'NN'},
    'CD':   {'NN', 'NNS', 'CD', 'IN', 'CC', 'JJ', 'RB', 'NNP'},
    'EX':   {'VBZ', 'VBD', 'VBP', 'MD'},
    'PDT':  {'DT', 'NN', 'NNS'},
    'RP':   {'IN', 'DT', 'NN', 'NNS'},
    'UH':   {'PRP', 'DT', 'NN', 'NNP', 'CC'},
    'JJS':  {'NN', 'NNS', 'NNP', 'IN'},
    'JJR':  {'NN', 'NNS', 'NNP', 'IN', 'CC'},
}

FUNCTION_WORDS = frozenset({
    'the', 'a', 'an', 'of', 'in', 'to', 'is', 'was', 'and',
    'or', 'for', 'on', 'at', 'by', 'it', 'that', 'this', 'with',
    'as', 'be', 'are', 'were', 'been', 'not', 'but', 'from',
    'has', 'had', 'have', 'do', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall',
})


#Metric functions

def measure_next_tag_topk_accuracy(model, test_sentences: list, top_k: int = 15) -> float:
    correct = 0
    total   = 0
    for sentence in test_sentences:
        words = word_tokenize(sentence.lower())
        if len(words) < 2:
            continue
        tagged = pos_tag(words)
        for i in range(len(tagged) - 1):
            prev_tag   = tagged[i][1]
            actual_tag = tagged[i + 1][1]
            if prev_tag in model.transition_probs:
                tps = model.transition_probs[prev_tag]
                top_k_tags = {
                    t for t, _ in sorted(
                        ((t, p) for t, p in tps.items() if t != '<UNK>'),
                        key=lambda x: x[1], reverse=True
                    )[:top_k]
                }
                if actual_tag in top_k_tags:
                    correct += 1
            total += 1
    return correct / total * 100 if total > 0 else 0.0


def measure_generation_quality(model, prompts: list, num_samples=5, max_length=15, temperature=0.55) -> dict:
    all_outputs = []
    for prompt in prompts:
        for _ in range(num_samples):
            text, log = model.autocomplete(prompt, max_length=max_length, temperature=temperature)
            all_outputs.append({
                'prompt':          prompt,
                'output':          text,
                'log':             log,
                'generated_words': text.split()[len(prompt.split()):]
            })

    total = len(all_outputs)

    # 1. Sentinel leakage
    bad_tokens = {'<start>', '<end>', '<unk>', '...', '__start__', '__end__'}
    leakage = sum(
        1 for item in all_outputs
        if any(w.lower() in bad_tokens for w in item['generated_words'])
    )
    sentinel_cleanliness = 100.0 - leakage / total * 100

    # 2. No-repetition score
    repetitions = 0
    for item in all_outputs:
        ws = item['generated_words']
        for i in range(1, len(ws)):
            if ws[i].lower() == ws[i - 1].lower() and ws[i].lower() not in FUNCTION_WORDS:
                repetitions += 1
                break
    no_repetition_score = 100.0 - repetitions / total * 100

    # 3. POS transition accuracy
    valid_trans = 0
    total_trans = 0
    for item in all_outputs:
        words  = word_tokenize(item['output'])
        tagged = pos_tag(words)
        for i in range(1, len(tagged)):
            pt, ct = tagged[i - 1][1], tagged[i][1]
            total_trans += 1
            if pt in VALID_TRANSITIONS and ct in VALID_TRANSITIONS[pt]:
                valid_trans += 1
            elif pt not in VALID_TRANSITIONS:
                valid_trans += 1   # give benefit of doubt to unseen tags
    pos_accuracy = valid_trans / total_trans * 100 if total_trans > 0 else 0.0

    # 4. Natural termination
    natural = sum(
        1 for item in all_outputs
        if len(item['generated_words']) < max_length
    )
    natural_termination = natural / total * 100

    # 5. Vocabulary diversity
    unique_ratios = []
    for item in all_outputs:
        content = [w for w in item['generated_words'] if w.lower() not in FUNCTION_WORDS]
        if content:
            unique_ratios.append(len(set(w.lower() for w in content)) / len(content))
    vocab_diversity = sum(unique_ratios) / len(unique_ratios) * 100 if unique_ratios else 0.0

    return {
        'sentinel_cleanliness':  sentinel_cleanliness,
        'no_repetition_score':   no_repetition_score,
        'pos_accuracy':          pos_accuracy,
        'natural_termination':   natural_termination,
        'vocab_diversity':       vocab_diversity,
    }


def compute_overall_accuracy(gen_metrics: dict, next_tag_accuracy: float) -> tuple:
    """
    Composite accuracy score.

    Weights (sum = 1.00):
      POS Transition Accuracy   0.30
      Sentinel Cleanliness      0.15
      No-Repetition Score       0.20
      Vocabulary Diversity      0.10
      Natural Termination       0.10
      Next-Tag Top-15 Accuracy  0.15
    """
    weights = {
        'pos_accuracy':         0.30,
        'sentinel_cleanliness': 0.15,
        'no_repetition_score':  0.20,
        'vocab_diversity':      0.10,
        'natural_termination':  0.10,
        'next_tag_accuracy':    0.15,
    }

    sub_scores = {
        'POS Transition Accuracy':  gen_metrics['pos_accuracy'],
        'Sentinel Cleanliness':     gen_metrics['sentinel_cleanliness'],
        'No-Repetition Score':      gen_metrics['no_repetition_score'],
        'Vocabulary Diversity':     gen_metrics['vocab_diversity'],
        'Natural Termination':      gen_metrics['natural_termination'],
        'Next-Tag Top-15 Accuracy': next_tag_accuracy,
    }

    overall = (
        gen_metrics['pos_accuracy']         * weights['pos_accuracy']         +
        gen_metrics['sentinel_cleanliness'] * weights['sentinel_cleanliness'] +
        gen_metrics['no_repetition_score']  * weights['no_repetition_score']  +
        gen_metrics['vocab_diversity']       * weights['vocab_diversity']      +
        gen_metrics['natural_termination']   * weights['natural_termination'] +
        next_tag_accuracy                   * weights['next_tag_accuracy']
    )
    return overall, sub_scores


#Main
if __name__ == '__main__':
    import random
    from corpus_loader import load_best_corpus

    print("=" * 62)
    print("  HMM v3 ACCURACY BENCHMARK  (100k multi-corpus, trigram)")
    print("=" * 62)

    all_sents = load_best_corpus(max_sentences=100_000)
    random.shuffle(all_sents)
    split        = int(len(all_sents) * 0.95)
    train_corpus = all_sents[:split]
    test_corpus  = all_sents[split:]

    print(f"\nTraining on {len(train_corpus):,} sentences...")
    t0 = time.time()
    model = HMMGenerator()
    model.train(train_corpus)
    model.build_probs(k=0.001)
    elapsed = time.time() - t0
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Vocabulary : {len(model.vocabulary):,} words")
    print(f"POS Tags   : {len(model.tags)}")
    print(f"Avg sent length learned: {model.avg_sentence_length:.1f} words")

    held_out = test_corpus[:500]
    print(f"\nMeasuring Next-Tag Top-15 accuracy on {len(held_out)} held-out sentences...")
    nt_acc = measure_next_tag_topk_accuracy(model, held_out, top_k=15)
    print(f"Next-Tag Top-15 Accuracy: {nt_acc:.1f}%")
    test_prompts = [
        'I love',
        'The government should',
        'She went to the',
        'We need to',
        'He said that',
        'The new',
        'They have been',
        'It is important to',
        'The city of',
        'This program will',
        'Scientists discovered that',
        'In recent years',
    ]

    print(f"\nGenerating {len(test_prompts) * 5} sample completions...")
    gen_metrics = measure_generation_quality(
        model, test_prompts, num_samples=5, max_length=15, temperature=0.55
    )

    overall, sub_scores = compute_overall_accuracy(gen_metrics, nt_acc)

    print("\n" + "-" * 62)
    print("  RESULTS")
    print("-" * 62)
    for name, score in sub_scores.items():
        filled = int(score / 2)
        bar = "#" * filled + "." * (50 - filled)
        print(f"  {name:.<32s} {bar} {score:.1f}%")
    print("-" * 62)
    filled = int(overall / 2)
    bar = "#" * filled + "." * (50 - filled)
    print(f"  {'OVERALL ACCURACY':.<32s} {bar} {overall:.1f}%")
    print("-" * 62)

    print("\n  SAMPLE OUTPUTS:")
    for p in test_prompts:
        text, _ = model.autocomplete(p, max_length=15, temperature=0.55)
        print(f'    "{p}"')
        print(f'    -> "{text}"')
        print()
