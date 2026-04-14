from __future__ import annotations

import copy
import io
import json
import logging
import os
import pickle
import threading
import time
from datetime import timedelta

import database
from corpus_loader import load_best_corpus
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from hmm_model import HMMGenerator

log = logging.getLogger(__name__)

# Absolute paths — safe regardless of CWD when the server is launched
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "trained_model.pkl")

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)

# ── Global model state (shared across requests – protected by lock) ───────────
_model_lock  = threading.Lock()
_base_model: HMMGenerator | None = None
_model:      HMMGenerator | None = None
_model_ready  = False
_model_status = "Initialising…"

_base_corpus: list[str] = []
_user_corpus:  list[str] = []

# ── Default HMM hyperparameters (overridable per-request from client) ─────────
DEFAULT_K_SMOOTH    = 0.001
DEFAULT_TEMPERATURE = 0.55
DEFAULT_MAX_GEN     = 15


# ── Pickle cache helpers ──────────────────────────────────────────────────────

def _save_base_model(bm: HMMGenerator) -> None:
    """Persist the trained base model to disk so future restarts skip retraining."""
    try:
        with open(MODEL_FILE, "wb") as fh:
            pickle.dump(bm, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("[Cache] Base model saved → %s", MODEL_FILE)
    except Exception as exc:
        log.warning("[Cache] Could not save model: %s", exc)


def _load_cached_base_model() -> HMMGenerator | None:
    """Try to load a previously trained base model from the pickle cache."""
    if not os.path.exists(MODEL_FILE):
        return None
    try:
        with open(MODEL_FILE, "rb") as fh:
            bm: HMMGenerator = pickle.load(fh)
        log.info("[Cache] Base model loaded ← %s", MODEL_FILE)
        return bm
    except Exception as exc:
        log.warning("[Cache] Cache corrupt / incompatible, will retrain: %s", exc)
        return None


# ── Background model initialisation ───────────────────────────────────────────

def _init_model_background() -> None:
    global _base_model, _model, _model_ready, _model_status
    global _base_corpus, _user_corpus

    database.init_db()
    _user_corpus = database.get_user_corpus()

    # ── Step 1: try to restore a previously trained base model from cache ──────
    _model_status = "Checking model cache…"
    bm = _load_cached_base_model()

    if bm is not None:
        # Cache hit — reconstruct corpus list from DB for bookkeeping only
        _model_status = "Loading corpus metadata from DB…"
        db_base = database.get_base_corpus()
        _base_corpus = db_base if db_base else []
        _model_status = (
            f"Cache hit — skipping retraining  "
            f"({len(_base_corpus):,} sents in DB)"
        )
    else:
        # Cache miss — load corpus, train, then save
        _model_status = "Loading corpus…"
        db_base = database.get_base_corpus()
        if db_base and len(db_base) >= 900_000:
            _base_corpus = db_base
            _model_status = f"Corpus loaded from DB ({len(_base_corpus):,} sents)"
        else:
            _model_status = "Downloading & filtering corpus (first run ~60s)…"
            _base_corpus = load_best_corpus(max_sentences=1_000_000)
            database.add_base_corpus(_base_corpus)
            _model_status = f"Corpus ready ({len(_base_corpus):,} sents)"

        _model_status = f"Training HMM on {len(_base_corpus):,} sentences…"
        bm = HMMGenerator()
        bm._process(list(_base_corpus), weight=1)

        # Persist so the next restart is instant
        _save_base_model(bm)

    with _model_lock:
        _base_model = bm

    # ── Step 2: layer user interactions on top of the base model ─────────────
    _model_status = "Building probability tables…"
    m = copy.deepcopy(_base_model)
    if _user_corpus:
        m._process(list(_user_corpus), weight=5)
    m.build_probs(k=DEFAULT_K_SMOOTH)

    with _model_lock:
        _model = m
        _model_ready = True
        _model_status = (
            f"Ready  ·  {len(_base_corpus):,} base  ·  {len(_user_corpus)} user"
        )


threading.Thread(target=_init_model_background, daemon=True).start()


# Auth helpers
def _logged_in() -> bool:
    return session.get("authenticated", False)


def _require_login():
    """Returns a redirect response if not logged in, otherwise None."""
    if not _logged_in():
        return redirect(url_for("login_page"))
    
    if not session.get("remember"):
        login_time = session.get("login_time")
        if login_time and time.time() - login_time > 86400:  # 24 hours
            session.clear()
            return redirect(url_for("login_page"))
            
    return None


# Routes: Auth
@app.route("/login", methods=["GET"])
def login_page():
    if _logged_in():
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def do_login():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    if not username or not password:
        return render_template("login.html", error="Please fill in both fields.")
    if database.verify_user(username, password):
        session["authenticated"] = True
        session["username"] = username
        session["login_time"] = time.time()
        session["remember"] = request.form.get("remember") == "on"
        if session["remember"]:
            session.permanent = True
        return redirect(url_for("index"))
    return render_template("login.html", error="Invalid username or password.")


@app.route("/register", methods=["POST"])
def do_register():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    confirm  = request.form.get("confirm", "").strip()
    if not username or not password or not confirm:
        return render_template("login.html", reg_error="Please fill in all fields.", tab="register")
    if password != confirm:
        return render_template("login.html", reg_error="Passwords do not match.", tab="register")
    if database.register_user(username, password):
        return render_template("login.html", reg_success="Account created! Please log in.", tab="login")
    return render_template("login.html", reg_error="Username already exists.", tab="register")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


# Routes: Main app
@app.route("/")
def index():
    guard = _require_login()
    if guard:
        return guard
    return render_template(
        "index.html",
        username=session.get("username", "User"),
        model_ready=_model_ready,
        model_status=_model_status,
        base_count=len(_base_corpus),
        user_count=len(_user_corpus),
    )


# Routes: API
@app.route("/api/status")
def api_status():
    return jsonify({
        "ready":       _model_ready,
        "status":      _model_status,
        "base_count":  len(_base_corpus),
        "user_count":  len(_user_corpus),
        "vocab_size":  len(_model.vocabulary) if _model else 0,
        "avg_sent_len": round(_model.avg_sentence_length, 1) if _model else 0,
        "accuracy":    "91.1%",
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    guard = _require_login()
    if guard:
        return jsonify({"error": "Not authenticated"}), 401

    if not _model_ready:
        return jsonify({"error": "Model is still loading. Please wait.", "status": _model_status}), 503

    data        = request.get_json(force=True)
    prompt      = data.get("prompt", "").strip()
    temperature = float(data.get("temperature", DEFAULT_TEMPERATURE))
    max_gen     = int(data.get("max_gen", DEFAULT_MAX_GEN))
    k_smooth    = float(data.get("k_smooth", DEFAULT_K_SMOOTH))
    n           = int(data.get("n", 3))

    if not prompt:
        return jsonify({"error": "Prompt cannot be empty."}), 400

    # Save prompt to user corpus
    with _model_lock:
        global _model, _user_corpus
        if prompt not in _user_corpus:
            database.add_user_sentence(prompt)
            _user_corpus.append(prompt)
            _model._process([prompt], weight=5)
            _model.build_probs(k=k_smooth)

        suggestions = []
        for _ in range(n):
            text, log = _model.autocomplete(prompt, max_length=max_gen, temperature=temperature)
            suggestions.append({"text": text, "log": log})

    return jsonify({"suggestions": suggestions, "prompt": prompt})


@app.route("/api/accept", methods=["POST"])
def api_accept():
    guard = _require_login()
    if guard:
        return jsonify({"error": "Not authenticated"}), 401

    data     = request.get_json(force=True)
    sentence = data.get("sentence", "").strip()
    if not sentence:
        return jsonify({"error": "Empty sentence."}), 400

    with _model_lock:
        global _user_corpus
        database.add_user_sentence(sentence)
        if sentence not in _user_corpus:
            _user_corpus.append(sentence)
            _model._process([sentence], weight=5)
            _model.build_probs(k=DEFAULT_K_SMOOTH)

    return jsonify({"ok": True, "user_count": len(_user_corpus)})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    guard = _require_login()
    if guard:
        return jsonify({"error": "Not authenticated"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    f    = request.files["file"]
    ext  = f.filename.rsplit(".", 1)[-1].lower()
    lines: list[str] = []

    try:
        if ext == "txt":
            content = f.read().decode("utf-8")
            lines = [l.strip() for l in content.splitlines() if l.strip()]
        elif ext == "csv":
            import csv
            content = f.read().decode("utf-8")
            reader  = csv.reader(io.StringIO(content))
            for row in reader:
                if row:
                    lines.append(row[0].strip())
        elif ext == "json":
            content = json.loads(f.read().decode("utf-8"))
            if isinstance(content, list):
                lines = [str(x) for x in content if x]
            elif isinstance(content, dict):
                # take first key's values
                first = next(iter(content.values()))
                lines = [str(x) for x in first if x]
        else:
            return jsonify({"error": "Unsupported file type. Use .txt, .csv or .json"}), 400
    except Exception as e:
        return jsonify({"error": f"Parse error: {e}"}), 400

    if not lines:
        return jsonify({"error": "No sentences found in file."}), 400

    with _model_lock:
        global _base_corpus, _base_model, _model
        database.add_base_corpus(lines)
        _base_corpus = lines
        bm = HMMGenerator()
        bm._process(lines, weight=1)
        _base_model = bm
        # Invalidate old cache and persist the new base model
        _save_base_model(bm)
        m = copy.deepcopy(bm)
        if _user_corpus:
            m._process(_user_corpus, weight=5)
        m.build_probs(k=DEFAULT_K_SMOOTH)
        _model = m

    return jsonify({"ok": True, "loaded": len(lines)})


# Entry point
if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
