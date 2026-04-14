# Synapse: HMM Autocomplete Engine

## 1. Project Overview

Synapse is a Streamlit-based AI autocomplete project that predicts sentence completions using a Hidden Markov Model (HMM). The system learns from two sources: a static base corpus built from the NLTK Brown Corpus and a dynamic user corpus that stores accepted or edited suggestions. The app also includes user authentication, persistent SQLite storage, interactive controls for generation behavior, and an explainability dashboard that shows how each suggestion was produced.

The main goal of the project is to demonstrate how classical probabilistic NLP can be turned into an interactive application with personalization, transparency, and feedback-driven learning.

## 2. What Each Folder/File Does

- `app.py`: Main Streamlit application. It loads the corpus, handles login gating, builds the model, renders the UI, manages user feedback, and shows the explainability dashboard.
- `hmm_model.py`: Core HMM engine. It tokenizes text, applies POS tagging, builds transition and emission probabilities, and generates autocomplete suggestions with temperature-based sampling.
- `database.py`: SQLite persistence layer. It stores the base corpus, user corpus, and authentication data.
- `login_ui.py`: Login and registration screen. It checks credentials and controls access to the main app.
- `ui.py`: UI helper functions. It loads CSS, color-codes POS tags, and renders the transition graph.
- `benchmark.py`: Evaluation script. It measures perplexity, POS transition quality, repetition, leakage, and overall generation quality.
- `test_model.py` and `test_model2.py`: Testing and debugging scripts for generation behavior and tag distribution.
- `style.css`: Visual styling for the Streamlit interface.
- `Untitled1.ipynb`: Notebook used for experimentation and model inspection.

## 3. Architecture and Data Flow

1. The app starts in `app.py` and initializes the database.
2. The login screen appears first through `login_ui.require_login()`.
3. Once authenticated, the base corpus is loaded from SQLite; if it is missing, the Brown Corpus subset is downloaded and stored.
4. The model is created using `HMMGenerator` from `hmm_model.py`.
5. The base corpus is trained with weight 1, while the user corpus is trained with higher weight 3 to prioritize feedback.
6. Probabilities are built using Laplace smoothing.
7. When the user enters a partial sentence, the model generates three suggestions.
8. Accept, reject, and edit actions update the user corpus and are saved back into the database.
9. The dashboard visualizes the reasoning trace, POS transition graph, and probability tables.

## 4. Implementation Details

### HMM Core Logic

The model uses POS tags rather than raw words to generate structurally valid completions. In `hmm_model.py`, each sentence is tokenized, wrapped with `<START>` and `<END>`, and tagged using NLTK `pos_tag()`. The code counts:

- transition counts between POS tags
- emission counts from POS tags to words
- start tag frequencies

These counts are later converted to probabilities in `build_probs()`.

### Laplace Smoothing

The model uses Laplace smoothing so unseen words or transitions do not get zero probability. This is important because autocomplete should still produce a result even when the prompt contains unfamiliar tokens.

### Temperature Sampling

Generation is not fully deterministic. The `temperature_sample()` function in `hmm_model.py` changes the randomness of selection:

- low temperature = more greedy and stable output
- medium temperature = balanced output
- high temperature = more creative but less predictable output

### Personalized Feedback Loop

The app stores accepted and edited suggestions in the user corpus. Those sentences are given more weight during training, so the model gradually adapts to user preferences.

### Authentication and Persistence

`database.py` creates three tables:

- `base_corpus` for the main training dataset
- `user_corpus` for saved suggestions and corrections
- `users` for login credentials

Passwords are stored as SHA-256 hashes, not plain text.

### Explainability Dashboard

The app does not only show the output sentence. It also exposes the internal reasoning:

- AI reasoning trace: shows the POS-by-POS generation path
- Transition graph: visualizes tag movement with NetworkX and Matplotlib
- Probability matrices: displays transition and emission probabilities

This is useful for demonstrating that the system is explainable, not a black box.

## 5. Performance and Limitations

The project already includes a note about a major performance issue: every Streamlit rerun can retrain large parts of the corpus and rebuild probability tables. That is accurate for the current implementation and is a good point to mention in the presentation.

The best improvement would be incremental updating instead of full retraining on every interaction, plus caching the model instance with Streamlit resource caching.

## 6. 7–8 Minute PPT Structure

### Slide 1: Title and Objective, 30 sec

Introduce the project, the problem it solves, and why HMM was chosen for autocomplete.

### Slide 2: Problem Statement, 45 sec

Explain the need for an autocomplete system that is lightweight, personalized, and explainable.

### Slide 3: System Architecture, 1 min

Show the folder structure and the flow from login to corpus loading, training, prediction, and feedback.

### Slide 4: HMM Working, 1 min

Explain POS tagging, transition probabilities, emission probabilities, and how the model generates the next word.

### Slide 5: Key Features, 1 min

Highlight Brown Corpus bootstrapping, user feedback learning, Laplace smoothing, temperature control, and authentication.

### Slide 6: UI and Explainability, 1 min

Show the Streamlit interface, action buttons, graph visualization, and reasoning trace.

### Slide 7: Testing and Benchmarking, 1 min

Mention perplexity, POS transition accuracy, repetition rate, and sentinel leakage checks from `benchmark.py`.

### Slide 8: Limitations and Future Work, 45 sec

Discuss retraining overhead, incremental learning as the next improvement, and possible extensions like n-gram hybrids or better caching.

## 7. Ready-to-Use PPT Prompt

Use this prompt if you want to generate slides with an AI PPT tool:

Create a 7 to 8 minute presentation on the project “Synapse: HMM Autocomplete Engine.” The presentation should explain the project in a clear technical but accessible way. Include these sections: title slide, problem statement, system architecture, HMM working principle, implementation details, UI and explainability features, evaluation and testing, and future improvements. The project is built with Streamlit, NLTK, SQLite, NetworkX, and Matplotlib. It uses the Brown Corpus as a base dataset, stores user feedback in SQLite, applies POS tagging for Hidden Markov Model based autocomplete, and supports Laplace smoothing and temperature-based sampling. Show that the app is interactive, personalized, and explainable. Mention that the model generates three autocomplete suggestions, supports accept/reject/edit feedback, and visualizes the reasoning trace, transition graph, and probability matrices. End with current limitations and improvement ideas such as incremental learning and caching.

## 8. Short Speaking Script

This project is an AI autocomplete system called Synapse. It uses a Hidden Markov Model to predict sentence completions based on part-of-speech transitions instead of simple word matching. The app starts with a login screen, loads a base corpus from the Brown Corpus, and then learns from user interactions stored in SQLite. The model uses Laplace smoothing to handle unseen cases and temperature sampling to control creativity. It generates three suggestions, and users can accept, reject, or edit them, which makes the model more personalized over time. The system also includes an explainability dashboard that shows the reasoning trace, transition graph, and probability tables, so the generation process is transparent. The main limitation is that training is still relatively heavy on reruns, so the next step is incremental updating and caching for better performance.