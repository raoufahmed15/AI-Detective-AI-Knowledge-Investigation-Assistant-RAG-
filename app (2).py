"""
AI Detective — AI Knowledge Investigation Assistant
====================================================

Streamlit front-end + serving layer for the RAG pipeline built in the
"mid-term-project" notebook.

Pipeline:
PDF incident reports + structured crime records
-> multilingual-e5-base embeddings
-> FAISS IndexFlatIP
-> Groq LLM answer generation
-> conversation memory + query rewriting

Groq API key is configured through environment variables or Streamlit secrets.
"""

# ============================================================================
# Imports
# ============================================================================

import json
import os
import pickle
import re

import requests
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer


# ============================================================================
# Fixed configuration
# ============================================================================

APP_TITLE = "AI Detective"
APP_SUBTITLE = "AI Knowledge Investigation Assistant"

ARTIFACTS_DIR = "model"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_REWRITE_MODEL = os.getenv("GEMINI_REWRITE_MODEL", GEMINI_MODEL)
GROQ_GENERATION_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_REWRITE_MODEL = os.getenv("GROQ_REWRITE_MODEL", GROQ_GENERATION_MODEL)

FALLBACK_GENERATION_MODELS = [
    GROQ_GENERATION_MODEL,
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
]

FALLBACK_REWRITE_MODELS = [
    GROQ_REWRITE_MODEL,
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
]

TOP_K = 5
# Minimum cosine score for a chunk to count as evidence.
# e5 scores cluster high, so calibrate this with the debug line below.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.80"))
DEBUG_SCORES = os.getenv("DEBUG_SCORES", "0") == "1"
TEMPERATURE = 0.5
SHOW_SOURCES = True

OUT_OF_SCOPE_TOKEN = "OUT_OF_SCOPE"
OUT_OF_SCOPE_MESSAGE = (
    "السؤال ده خارج نطاق قاعدة البيانات. "
    "أنا متخصص في القضايا والحوادث الموجودة في التقارير والسجلات فقط، "
    "اسألني عن حادثة أو رقم قضية (مثل INC-001)."
)
GREETING_MESSAGE = (
    "أهلاً! أنا AI Detective. اسألني عن القضايا والحوادث الموجودة في "
    "التقارير والسجلات."
)


# ============================================================================
# Page configuration
# ============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🕵️",
    layout="centered",
)

# Current google-genai releases support the 2026 Gemini authentication changes.
# Streamlit Cloud should install google-genai==2.24.0 from requirements.txt.


# ============================================================================
# Styling
# ============================================================================

st.markdown(
    """
    <style>

    :root {
        --bg: #0b1220;
        --bg-soft: #111b2d;
        --panel: rgba(18, 27, 39, 0.82);
        --panel-strong: rgba(21, 30, 43, 0.98);
        --line: rgba(148, 163, 184, 0.16);
        --text: #edf6ff;
        --muted: #a5b4c9;
        --primary: #ff5a36;
        --secondary: #ffb347;
        --success: #59d38b;
        --shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 90, 54, 0.18), transparent 24%),
            radial-gradient(circle at top right, rgba(87, 129, 255, 0.14), transparent 26%),
            linear-gradient(180deg, #070d18 0%, #0d1524 100%);
        color: var(--text);
    }

    .block-container {
        padding-top: 2.25rem;
        padding-bottom: 2rem;
        max-width: 960px;
    }

    .ai-detective-header {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        padding: 1rem 1.15rem;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(19, 30, 46, 0.94), rgba(11, 18, 32, 0.9));
        box-shadow: var(--shadow);
        margin-bottom: 1.25rem;
    }

    .ai-detective-header h1 {
        margin: 0;
        font-size: 1.9rem;
        letter-spacing: -0.04em;
        color: var(--text);
    }

    .ai-detective-header p {
        margin: 0.3rem 0 0;
        opacity: 0.8;
        font-size: 0.92rem;
        color: var(--muted);
    }

    .stChatMessage {
        font-size: 0.97rem;
        line-height: 1.6;
        color: var(--text);
    }

    .stChatMessage > div {
        border-radius: 18px !important;
        border: 1px solid var(--line) !important;
        box-shadow: 0 8px 20px rgba(7, 11, 18, 0.25);
        backdrop-filter: blur(2px);
    }

    .stChatMessage[data-testid="stChatMessageUser"] > div {
        background: linear-gradient(180deg, rgba(24, 35, 49, 0.96), rgba(14, 22, 31, 0.94));
        border-color: rgba(95, 120, 164, 0.25) !important;
    }

    .stChatMessage[data-testid="stChatMessageAssistant"] > div {
        background: linear-gradient(180deg, rgba(255, 153, 89, 0.16), rgba(255, 90, 54, 0.08));
        border-color: rgba(255, 123, 76, 0.22) !important;
    }

    .stChatInput {
        border-radius: 18px !important;
        border: 1px solid rgba(96, 105, 125, 0.3) !important;
        background: rgba(15, 23, 35, 0.88) !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.28);
    }

    .stChatInput textarea {
        background: transparent !important;
        color: var(--text) !important;
        font-size: 0.98rem !important;
    }

    .stChatInput button {
        border-radius: 12px !important;
        background: linear-gradient(135deg, var(--primary), #ff7c45) !important;
        color: white !important;
        font-weight: 700 !important;
    }

    .evidence-card {
        border: 1px solid rgba(140, 160, 190, 0.2);
        border-radius: 14px;
        padding: 0.8rem 0.9rem;
        margin-bottom: 0.5rem;
        font-size: 0.82rem;
        background: rgba(15, 23, 35, 0.78);
        color: var(--text);
    }

    .evidence-score {
        display: inline-block;
        padding: 0.08rem 0.6rem;
        border-radius: 999px;
        background: rgba(89, 211, 139, 0.12);
        border: 1px solid rgba(89, 211, 139, 0.18);
        color: #aef0c6;
        font-weight: 700;
        font-size: 0.72rem;
    }

    .stExpander {
        border: 1px solid var(--line) !important;
        border-radius: 14px !important;
        background: rgba(13, 24, 37, 0.74) !important;
    }

    .stExpander summary {
        color: var(--text) !important;
        font-weight: 600;
    }

    .sidebar-content {
        background: linear-gradient(180deg, rgba(10, 17, 27, 0.97), rgba(12, 21, 32, 0.9));
    }

    code {
        background: rgba(148, 163, 184, 0.12) !important;
        color: #ffdf99 !important;
        border-radius: 6px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Model API credentials
# ============================================================================

def get_api_credentials():
    """Return credentials paired with the provider that issued them."""

    try:
        secrets = st.secrets
    except Exception:
        secrets = {}

    gemini_key = os.getenv("GEMINI_API_KEY") or secrets.get("GEMINI_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY") or secrets.get("GROQ_API_KEY", "")

    if gemini_key and str(gemini_key).strip():
        return "gemini", str(gemini_key).strip()
    if groq_key and str(groq_key).strip():
        return "groq", str(groq_key).strip()

    st.error(
        "Model API key is missing. Configure GEMINI_API_KEY or GROQ_API_KEY "
        "in the environment or Streamlit secrets."
    )
    st.stop()


@st.cache_resource
def load_gemini_client(api_key):
    from google import genai

    return genai.Client(api_key=api_key)

# ============================================================================
# Load artifacts
# ============================================================================

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_artifacts(directory: str):

    config_path = os.path.join(directory, "config.json")
    index_path = os.path.join(directory, "index.faiss")
    metadata_path = os.path.join(directory, "metadata.pkl")

    for path in (config_path, index_path, metadata_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    if index.ntotal != len(metadata):
        raise ValueError(
            f"FAISS index contains {index.ntotal} vectors but metadata contains "
            f"{len(metadata)} records. Rebuild the model artifacts."
        )

    return config, index, metadata


# ============================================================================
# Embedding model
# ============================================================================

@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_embedding_model(model_name: str):

    return SentenceTransformer(
        model_name,
        device="cpu",
    )


# ============================================================================
# RAG retrieval
# ============================================================================

def search_index(
    query,
    model,
    index,
    metadata,
    top_k=5,
    min_score=MIN_SCORE,
):
    """
    Search the FAISS index using multilingual-e5-base.
    """

    q = model.encode(
        [f"query: {query}"],
        convert_to_numpy=True,
    )

    faiss.normalize_L2(q)

    incident_match = re.search(r"\bINC-\d+\b", str(query), re.IGNORECASE)
    search_count = index.ntotal if incident_match else top_k
    scores, ids = index.search(q, search_count)

    results = []

    for score, idx in zip(scores[0], ids[0]):

        if idx == -1:
            continue

        results.append(
            {
                **metadata[idx],
                "score": float(score),
            }
        )

    if DEBUG_SCORES:
        st.sidebar.write("scores:", [round(r["score"], 3) for r in results[:top_k]])

    if incident_match:
        incident_id = incident_match.group(0).upper()
        results.sort(
            key=lambda item: (
                str(item.get("incident_id", "")).upper() != incident_id,
                -item["score"],
            )
        )
        # An explicit INC-xxx lookup is always kept; everything else needs a real score.
        results = [
            r for r in results
            if r["score"] >= min_score
            or str(r.get("incident_id", "")).upper() == incident_id
        ]
    else:
        results = [r for r in results if r["score"] >= min_score]

    return results[:top_k]


# ============================================================================
# Conversation context
# ============================================================================

def build_context(
    history,
    max_turns=6,
):
    """
    Convert conversation history into plain text.
    """

    return "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}"
        for turn in history[-max_turns:]
    )


# ============================================================================
# RAG prompt
# ============================================================================

def build_rag_prompt(
    question,
    evidence,
    history_text,
    incident_ids,
):

    evidence_block = "\n\n".join(
        f"[{i}] "
        f"({e['source_name']}, "
        f"{'page ' + str(e['page']) if e['source_type'] == 'pdf' else 'row ' + str(e['row_number'])}, "
        f"incident {e['incident_id']}, "
        f"score {e['score']:.2f})\n"
        f"{e['text']}"
        for i, e in enumerate(evidence, 1)
    )

    if not evidence_block:
        evidence_block = "(no evidence retrieved)"

    return f"""
You are AI Detective.

Answer the user's actual question directly. Use retrieved evidence for factual
claims about incidents; do not merely copy a retrieved passage.

Rules:

- Never invent facts.
- Answer ONLY from the retrieved evidence and the incident list. Never use outside
    general knowledge (sports, celebrities, news, etc.).
- If the question is not about the incidents/records in the database, or the
    retrieved evidence is not relevant to it, reply with exactly the single word
    OUT_OF_SCOPE and nothing else. Never offer to help with other topics.
- If asked which incidents are in the database, use the available incident IDs below.
- For an investigative question, if evidence is insufficient, say so clearly.
- A shared detail across cases is only a POSSIBLE connection, never proof.
- Do not treat similarity as proof of identity or causation.
- Mention relevant incident IDs when synthesizing multiple sources.
- Give a useful, specific answer with enough detail to address the question.
- Answer in the same language as the user's question.
- Prefer direct evidence over assumptions.
- If multiple pieces of evidence conflict, explicitly mention the conflict.

Available incident IDs:
{", ".join(incident_ids) or "none"}

Conversation so far:
{history_text or "(none)"}

Retrieved evidence:
{evidence_block}

Question:
{question}

Answer:
""".strip()


# ============================================================================
# Model client
# ============================================================================

def get_client():
    """Create a client using credentials for their matching provider."""

    provider, api_key = get_api_credentials()
    client = {"api_key": api_key, "provider": provider}
    if provider == "gemini":
        client["client"] = load_gemini_client(api_key)
    return client


# ============================================================================
# Input safety helpers
# ============================================================================

def is_abusive_or_empty_request(question):
    """Reject unsafe or empty prompts before they hit the rewrite/model pipeline."""

    if question is None:
        return True

    cleaned = str(question).strip()
    if not cleaned:
        return True

    normalized = cleaned.lower()
    abusive_terms = (
        "go to hell",
        "idiot",
        "stupid",
        "dumbass",
        "hate you",
        "kill yourself",
        "fuck",
        "f***",
        "bitch",
        "asshole",
        "damn",
    )

    return any(term in normalized for term in abusive_terms)


def extract_text_from_content(content):
    """Normalize Groq/OpenAI content payloads that may be a string or a list."""

    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        fragments = []
        for item in content:
            if isinstance(item, str):
                fragments.append(item)
            elif isinstance(item, dict):
                for key in ("text", "content"):
                    value = item.get(key)
                    if isinstance(value, str):
                        fragments.append(value)
                    elif isinstance(value, list):
                        fragments.extend(
                            part for part in (extract_text_from_content(value) for value in value) if part
                        )
        return "\n".join(fragments).strip()

    if isinstance(content, dict):
        for key in ("text", "content"):
            value = content.get(key)
            if value:
                return extract_text_from_content(value)

    return ""


# ============================================================================
# Groq text generation
# ============================================================================

def generate_text(
    client,
    prompt,
    model,
    max_tokens=500,
    temperature=0.5,
    fallback_models=None,
):
    """Generate text with the provider paired to the configured API key."""

    api_key = client["api_key"]
    if client["provider"] == "gemini":
        response = client["client"].interactions.create(
            model=model or GEMINI_MODEL,
            input=prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        text = str(response.output_text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty response.")
        return text

    url = "https://api.groq.com/openai/v1/chat/completions"
    candidate_models = []

    if model:
        candidate_models.append(model)

    if fallback_models:
        for fallback_model in fallback_models:
            if fallback_model and fallback_model not in candidate_models:
                candidate_models.append(fallback_model)

    last_error = None

    for candidate in candidate_models:
        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": candidate,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=60,
            )

            if not response.ok:
                try:
                    details = response.json()
                except Exception:
                    details = response.text
                last_error = RuntimeError(f"Groq API error {response.status_code}: {details}")
                continue

            payload = response.json()
            choices = payload.get("choices") or []
            if not choices:
                last_error = RuntimeError("Groq API returned no choices.")
                continue

            message = choices[0].get("message") or choices[0].get("delta") or {}
            text = extract_text_from_content(message.get("content"))
            if not text:
                last_error = RuntimeError("Groq API returned an empty response.")
                continue

            return str(text).strip()

        except Exception as error:
            last_error = error
            continue

    raise last_error or RuntimeError("Groq API failed to generate a response.")


# ============================================================================
# Query contextualization
# ============================================================================

def contextualize_query(
    client,
    history,
    question,
):

    if not history:
        return question

    if is_abusive_or_empty_request(question):
        return question

    prompt = f"""
Conversation so far:
{build_context(history, 4)}

New question:
"{question}"

Rewrite the new question as ONE standalone question.

Resolve pronouns such as:
- it
- that
- this
- they
- them
- he
- she

Use the previous conversation only when necessary.

Reply with ONLY the rewritten question.

Rewritten question:
""".strip()

    try:
        rewritten = generate_text(
            client=client,
            prompt=prompt,
            model=(GEMINI_REWRITE_MODEL if client["provider"] == "gemini" else GROQ_REWRITE_MODEL),
            max_tokens=60,
            temperature=0.3,
            fallback_models=FALLBACK_REWRITE_MODELS,
        )
    except Exception:
        return question

    rewritten = rewritten.split("\n")[0].strip()

    return rewritten or question


# ============================================================================
# Conversational input routing
# ============================================================================

def is_casual_conversation(question):
    """Avoid attaching unrelated case evidence to greetings and small talk."""

    normalized = str(question).strip().lower()
    normalized = re.sub(r"[\u064B-\u065F\u0670]", "", normalized)
    normalized = normalized.translate(str.maketrans("أإآ", "ااا"))

    greeting_pattern = r"^(اهلا|مرحبا|السلام عليكم|صباح الخير|مساء الخير|hello|hi)(?:\b|$)"
    if re.match(greeting_pattern, normalized):
        return True

    return False


def is_case_list_question(question):
    normalized = str(question).strip().lower()
    return any(
        term in normalized
        for term in ("القضايا", "الحوادث", "list cases", "show cases", "what cases")
    )


def rag_answer(
    client,
    question,
    history,
    index,
    embed_model,
    metadata,
    top_k=TOP_K,
    temperature=TEMPERATURE,
):

    if is_abusive_or_empty_request(question):
        fallback = (
            "I can’t help with abusive or non-investigative requests. "
            "Please ask a question about the case evidence or incident records."
        )
        return {
            "answer": fallback,
            "original_question": question,
            "contextualized_question": question,
            "sources": [],
        }

    # ------------------------------------------------------------
    # 1. Rewrite question using conversation history
    # ------------------------------------------------------------

    casual_conversation = is_casual_conversation(question)
    case_list_question = is_case_list_question(question)
    contextualized = question

    if casual_conversation:
        return {
            "answer": GREETING_MESSAGE,
            "original_question": question,
            "contextualized_question": question,
            "generation_error": None,
            "sources": [],
        }
    if history and not casual_conversation and not case_list_question:
        contextualized = contextualize_query(client, history, question)

    # ------------------------------------------------------------
    # 2. Retrieve evidence
    # ------------------------------------------------------------

    evidence = []
    if not casual_conversation and not case_list_question:
        evidence = search_index(
            contextualized,
            embed_model,
            index,
            metadata,
            top_k=top_k,
        )

    if not casual_conversation and not case_list_question and not evidence:
        return {
            "answer": (
                "مفيش أدلة في قاعدة البيانات ليها علاقة بسؤالك. "
                "أنا متخصص في القضايا والحوادث الموجودة في التقارير والسجلات فقط."
            ),
            "original_question": question,
            "contextualized_question": contextualized,
            "generation_error": None,
            "sources": [],
        }

    # ------------------------------------------------------------
    # 3. Build grounded RAG prompt
    # ------------------------------------------------------------

    prompt = build_rag_prompt(
        question,
        evidence,
        build_context(history),
        sorted(
            {
                str(item.get("incident_id", "")).upper()
                for item in metadata
                if re.fullmatch(r"INC-\d+", str(item.get("incident_id", "")).upper())
            }
        ),
    )

    # ------------------------------------------------------------
    # 4. Generate final answer
    # ------------------------------------------------------------

    generation_error = None
    try:
        answer = generate_text(
            client=client,
            prompt=prompt,
            model=(GEMINI_MODEL if client["provider"] == "gemini" else GROQ_GENERATION_MODEL),
            max_tokens=500,
            temperature=temperature,
            fallback_models=(FALLBACK_GENERATION_MODELS if client["provider"] == "groq" else None),
        )
    except Exception as error:
        answer = ""
        error_text = str(error)
        api_key = client.get("api_key")
        if api_key:
            error_text = error_text.replace(api_key, "[REDACTED]")
        generation_error = f"{type(error).__name__}: {error_text[:300]}"

    if answer.strip().upper().startswith(OUT_OF_SCOPE_TOKEN):
        return {
            "answer": OUT_OF_SCOPE_MESSAGE,
            "original_question": question,
            "contextualized_question": contextualized,
            "generation_error": None,
            "sources": [],
        }

    # ------------------------------------------------------------
    # 5. Update conversation memory
    # ------------------------------------------------------------

    if not generation_error:
        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )

    # ------------------------------------------------------------
    # 6. Return result
    # ------------------------------------------------------------

    return {
        "answer": answer,
        "original_question": question,
        "contextualized_question": contextualized,
        "generation_error": generation_error,
        "sources": [
            {
                key: evidence_item[key]
                for key in (
                    "source_type",
                    "source_name",
                    "page",
                    "row_number",
                    "incident_id",
                    "score",
                )
            }
            for evidence_item in evidence
        ],
    }


# ============================================================================
# Header
# ============================================================================

st.markdown(
    f"""
    <div class="ai-detective-header">
        <div style="font-size:2.2rem;">
            🕵️
        </div>
        <div>
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Startup checks
# ============================================================================

try:

    config, index, metadata = load_artifacts(
        ARTIFACTS_DIR
    )

except (FileNotFoundError, ValueError) as error:

    st.error(f"Setup error: {error}")

    st.info(
        f"Make sure the `{ARTIFACTS_DIR}/` folder contains:\n\n"
        "- config.json\n"
        "- index.faiss\n"
        "- metadata.pkl"
    )

    st.stop()


# ============================================================================
# Load embedding model
# ============================================================================

embed_model = load_embedding_model(
    config.get(
        "embedding_model",
        "intfloat/multilingual-e5-base",
    )
)


# ============================================================================
# Initialize Groq client
# ============================================================================

try:

    client = get_client()

except Exception as error:

    st.error(
        f"Groq initialization failed: {error}"
    )
    st.stop()


# ============================================================================
# Chat state
# ============================================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "rag_history" not in st.session_state:

    st.session_state.rag_history = []


# ============================================================================
# Empty state
# ============================================================================

if not st.session_state.messages:

    st.caption(
        "اسأل عن أي حادثة أو تفصيلة، "
        "وهيجاوبك بناءً على الأدلة المسترجعة فقط."
    )


# ============================================================================
# Render previous messages
# ============================================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message["content"]:
            st.markdown(message["content"])

        if message.get("generation_error"):
            st.warning(f"تعذر توليد الرد. التفاصيل: {message['generation_error']}")

        if (
            message["role"] == "assistant"
            and SHOW_SOURCES
            and message.get("sources")
        ):

            with st.expander(
                f"📎 {len(message['sources'])} source(s) used"
            ):

                for source in message["sources"]:

                    if source["source_type"] == "pdf":
                        location = f"page {source['page']}"
                    else:
                        location = f"row {source['row_number']}"

                    st.markdown(
                        f"""
                        <div class="evidence-card">

                            <span class="evidence-score">
                                score {source['score']:.2f}
                            </span>

                            &nbsp;

                            <b>{source['source_name']}</b>

                            ({location})

                            —

                            incident
                            <code>{source['incident_id']}</code>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ============================================================================
# Chat input
# ============================================================================

question = st.chat_input(
    "اكتب سؤالك هنا..."
)


# ============================================================================
# Handle question
# ============================================================================

if question:

    # ------------------------------------------------------------
    # Add user message
    # ------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # ------------------------------------------------------------
    # Generate assistant response
    # ------------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("جاري البحث..."):

            try:

                result = rag_answer(
                    client=client,
                    question=question,
                    history=st.session_state.rag_history,
                    index=index,
                    embed_model=embed_model,
                    metadata=metadata,
                    top_k=TOP_K,
                    temperature=TEMPERATURE,
                )

            except Exception as error:
                error_text = str(error)
                api_key = client.get("api_key")
                if api_key:
                    error_text = error_text.replace(api_key, "[REDACTED]")
                result = {
                    "answer": "",
                    "original_question": question,
                    "contextualized_question": question,
                    "generation_error": f"{type(error).__name__}: {error_text[:300]}",
                    "sources": [],
                }

        # --------------------------------------------------------
        # Show answer
        # --------------------------------------------------------

        if result["answer"]:
            st.markdown(result["answer"])

        if result.get("generation_error"):
            st.warning(
                "تعذر توليد الرد من نموذج الذكاء الاصطناعي. "
                f"التفاصيل: {result['generation_error']}"
            )

        # --------------------------------------------------------
        # Show sources
        # --------------------------------------------------------

        if (
            SHOW_SOURCES
            and result["sources"]
        ):

            with st.expander(
                f"📎 {len(result['sources'])} source(s) used"
            ):

                for source in result["sources"]:

                    if source["source_type"] == "pdf":
                        location = f"page {source['page']}"
                    else:
                        location = f"row {source['row_number']}"

                    st.markdown(
                        f"""
                        <div class="evidence-card">

                            <span class="evidence-score">
                                score {source['score']:.2f}
                            </span>

                            &nbsp;

                            <b>{source['source_name']}</b>

                            ({location})

                            —

                            incident
                            <code>{source['incident_id']}</code>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # ------------------------------------------------------------
    # Save assistant message
    # ------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "generation_error": result.get("generation_error"),
            "sources": result["sources"],
        }
    )
