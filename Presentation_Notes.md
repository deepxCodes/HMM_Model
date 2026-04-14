# Synapse Project Presentation Notes

## Project Summary

Synapse is a Streamlit-based autocomplete application that uses a Hidden Markov Model to generate sentence completions. It is designed as a practical demonstration of classical NLP, user feedback learning, and explainable AI behavior in one system. Instead of treating autocomplete as simple text prediction, the project models part-of-speech transitions and word emissions so that generated sentences follow a more grammatical structure.

The project also supports personalization. When a user accepts or edits a suggestion, that sentence is stored in SQLite and used again in later training. This means the system becomes more adapted to the user over time.

## What The Project Contains

The workspace is organized around a small but complete application stack:

- `app.py` is the main Streamlit app and the entry point for the whole system.
- `hmm_model.py` contains the HMM training and generation logic.
- `database.py` manages SQLite storage for the corpora and user authentication.
- `login_ui.py` displays the login and registration flow before the main app opens.
- `ui.py` contains shared UI helpers such as CSS loading, POS color coding, and graph rendering.
- `benchmark.py` evaluates output quality using metrics like perplexity and POS transition accuracy.
- `test_model.py` and `test_model2.py` are testing and debugging scripts used to verify model behavior.
- `style.css` contains the visual styling for the interface.
- `Untitled1.ipynb` is a notebook for experimentation and inspection.

## How The Application Works

The flow begins in `app.py`. The app initializes the database and then calls the login screen from `login_ui.py`. If the user is authenticated, the application loads the corpus and prepares the model.

The corpus comes from two sources:

1. A base corpus, which is built from the Brown Corpus and stored in SQLite.
2. A user corpus, which stores accepted or edited suggestions from the session history.

The model is then trained on both corpora, with the user corpus given more importance so the system can learn from feedback.

Once the model is ready, the user enters a partial sentence. The app generates three completions. Each suggestion can be accepted, rejected, or edited. When a suggestion is accepted or corrected, it is written back to the user corpus and stored in the database. That makes the system progressively more personalized.

## Implementation Details

### 1. Hidden Markov Model Logic

The HMM is implemented in `hmm_model.py`. It works on part-of-speech tags rather than only raw words. Every sentence is:

- tokenized into words
- wrapped with `<START>` and `<END>` markers
- passed through NLTK POS tagging

From this tagged sequence, the model builds three kinds of frequency tables:

- start counts for the first tag in a sentence
- transition counts between tags
- emission counts from tags to actual words

These counts are later normalized into probabilities.

### 2. Laplace Smoothing

The project applies Laplace smoothing to avoid zero-probability cases. This matters because autocomplete must still produce a result even if a word or tag has not been seen before. Smoothing makes the model more stable and avoids hard failures during generation.

### 3. Temperature-Based Sampling

The `temperature_sample()` function changes how random or deterministic the generation process is.

- Low temperature makes the model conservative and more repeatable.
- Medium temperature balances structure and variation.
- High temperature increases creativity but can reduce grammatical quality.

This is useful in a live presentation because you can demonstrate how the same prompt produces different outputs at different temperatures.

### 4. Database and Authentication

`database.py` creates three SQLite tables:

- `base_corpus` for the core training set
- `user_corpus` for saved feedback sentences
- `users` for login credentials

Passwords are stored as SHA-256 hashes. The login flow in `login_ui.py` checks credentials before the main UI is shown.

### 5. UI and Visualization

The app is not just a text generator. It also includes visualization and explainability features:

- The generated sentence is color coded by part of speech.
- A transition graph shows how tags move through the model.
- Probability tables expose the transition and emission values.
- The reasoning trace shows how each suggestion was built step by step.

These features make the model easier to present because you can explain not just what it generated, but why it generated it.

### 6. Performance Considerations

The project currently retrains or rebuilds parts of the model during Streamlit reruns, which can slow the app when the corpus grows. The best future improvement would be incremental updating and model caching so only the new feedback is processed instead of the full dataset.

## How To Present It Efficiently

### Opening Explanation

Start with the problem: autocomplete should not only be fast, it should also be grammatical, adaptable, and explainable. Then introduce Synapse as a Hidden Markov Model-based solution that learns from both a base corpus and user feedback.

### Architecture Explanation

Show the folder structure and explain that the project is divided into UI, model, database, and testing layers. This helps the audience understand that the app is modular rather than a single script.

### Core Technical Explanation

Explain that the model uses POS tags. Instead of predicting raw words directly, it predicts grammatical structure first and then selects words that match the predicted tag. This is the main reason the output feels more structured than a simple word-frequency generator.

### User Interaction Explanation

Point out the feedback loop. When the user accepts or edits a suggestion, the model stores that sentence and uses it for future learning. This is what makes the app personalized.

### Evaluation Explanation

Mention that the project is not just a demo. It includes benchmark scripts that check perplexity, repetition, leakage, and POS transition quality. That gives the presentation credibility because the system has measurable evaluation, not only visuals.

## Suggested Presentation Flow

1. Introduce the problem and project goal.
2. Explain the folder structure and architecture.
3. Describe the HMM implementation with POS tagging.
4. Explain Laplace smoothing and temperature sampling.
5. Demonstrate the UI, feedback loop, and explainability dashboard.
6. Mention testing and benchmarks.
7. Close with limitations and future improvements.

## One-Minute Closing Summary

Synapse is a personalized, explainable autocomplete system built with Streamlit, NLTK, and SQLite. It uses a Hidden Markov Model to generate structured sentence completions, adapts through user feedback, and presents its reasoning through graphs and probability tables. The main future improvement is to reduce retraining cost through incremental updates and caching.