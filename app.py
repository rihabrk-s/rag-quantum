import streamlit as st
import os
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum AI — RAG Assistant",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Sora:wght@300;400;600;700&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: #050a14 !important;
    color: #e2e8f4 !important;
}

/* ── Animated starfield background ── */
.stApp {
    background:
        radial-gradient(ellipse at 20% 50%, rgba(0,200,255,0.04) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(120,80,255,0.06) 0%, transparent 50%),
        #050a14 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #060e1e 100%) !important;
    border-right: 1px solid rgba(0,200,255,0.12) !important;
}
[data-testid="stSidebar"] * { color: #c8d8f0 !important; }

/* ── Main header ── */
.quantum-header {
    text-align: center;
    padding: 2rem 0 1rem;
    position: relative;
}
.quantum-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #00c8ff 0%, #7b5cfa 50%, #00ffb3 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.quantum-header .subtitle {
    font-size: 0.85rem;
    color: #5a7a9e;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 0.4rem;
    font-family: 'Space Mono', monospace;
}
.atom-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 0.5rem;
    animation: spin 12s linear infinite;
    filter: drop-shadow(0 0 16px rgba(0,200,255,0.5));
}
@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.4rem 0 !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"],
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    background: transparent !important;
}

/* Message bubbles via markdown container */
.user-msg, .bot-msg {
    padding: 1rem 1.3rem;
    border-radius: 16px;
    margin: 0.3rem 0;
    line-height: 1.65;
    font-size: 0.95rem;
}
.user-msg {
    background: linear-gradient(135deg, rgba(0,200,255,0.1), rgba(123,92,250,0.1));
    border: 1px solid rgba(0,200,255,0.2);
    border-radius: 16px 16px 4px 16px;
}
.bot-msg {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px 16px 16px 4px;
}

/* ── Chat input ── */
[data-testid="stChatInputContainer"] {
    background: rgba(10,22,40,0.9) !important;
    border-top: 1px solid rgba(0,200,255,0.1) !important;
    padding: 1rem !important;
}
[data-testid="stChatInputTextArea"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(0,200,255,0.2) !important;
    border-radius: 12px !important;
    color: #e2e8f4 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInputTextArea"]:focus {
    border-color: rgba(0,200,255,0.5) !important;
    box-shadow: 0 0 0 2px rgba(0,200,255,0.1) !important;
}

/* ── Source chips ── */
.source-chip {
    display: inline-block;
    background: rgba(0,200,255,0.08);
    border: 1px solid rgba(0,200,255,0.2);
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    color: #00c8ff;
    margin: 0.15rem;
}

/* ── Metrics cards ── */
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    margin: 0.3rem 0;
}
.metric-card .val {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #00c8ff;
}
.metric-card .lbl {
    font-size: 0.72rem;
    color: #4a6880;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,200,255,0.2); border-radius: 3px; }

/* ── Buttons ── */
.stButton button {
    background: linear-gradient(135deg, rgba(0,200,255,0.15), rgba(123,92,250,0.15)) !important;
    border: 1px solid rgba(0,200,255,0.3) !important;
    color: #e2e8f4 !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.85rem !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    border-color: rgba(0,200,255,0.6) !important;
    box-shadow: 0 0 12px rgba(0,200,255,0.2) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Credentials (from Streamlit secrets or env) ──────────────────────────────
def get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

GROQ_API_KEY    = get_secret("GROQ_API_KEY")
QDRANT_URL      = get_secret("QDRANT_URL")
QDRANT_API_KEY  = get_secret("QDRANT_API_KEY")
COLLECTION_NAME = get_secret("COLLECTION_NAME", "gots_chunks")


# ─── Init clients (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("BAAI/bge-m3")

@st.cache_resource(show_spinner=False)
def get_qdrant():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

@st.cache_resource(show_spinner=False)
def get_groq():
    return Groq(api_key=GROQ_API_KEY)


# ─── RAG core ─────────────────────────────────────────────────────────────────
def rag_query(question: str, top_k: int = 5, history: list = None) -> tuple[str, list]:
    embedder = load_embedder()
    qdrant   = get_qdrant()
    groq     = get_groq()

    q_emb = embedder.encode(question).tolist()
    results = qdrant.search(collection_name=COLLECTION_NAME, query_vector=q_emb, limit=top_k)

    if not results:
        return "Je n'ai pas trouvé d'information pertinente dans la base de connaissances.", []

    context = "\n\n---\n\n".join(
        f"[Chunk {r.payload['chunk_id']} | score: {r.score:.2f}]\n{r.payload['text']}"
        for r in results
    )
    sources = [
        {"chunk_id": r.payload["chunk_id"], "score": round(r.score, 3), "text": r.payload["text"][:200]}
        for r in results
    ]

    messages = [
        {
            "role": "system",
            "content": """You are an expert in Quantum AI — the intersection of quantum computing and artificial intelligence.
You have access to a scientific white paper (arXiv:2505.23860v3) on this topic.

STRICT RULES:
- Answer ONLY from the provided context
- If the answer is not in the context, reply exactly: "Je n'ai pas trouvé cette information dans la base de connaissances."
- Cite chunk numbers [Chunk X] when referencing a source
- Distinguish timelines: short-term (3-5 yrs) / mid-term (5-10 yrs) / long-term (>10 yrs) when relevant
- Structure: Summary → Details → Source(s)
- Reply in the language of the question (French, English, or Arabic)
- Be precise, scientific, and clear""",
        }
    ]

    # Inject conversation history
    if history:
        for h in history[-6:]:  # last 3 exchanges
            messages.append({"role": h["role"], "content": h["content"]})

    messages.append({
        "role": "user",
        "content": f"Context from white paper:\n\n{context}\n\nQuestion: {question}",
    })

    response = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )

    return response.choices[0].message.content, sources


# ─── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:2rem'>⚛️</div>
        <div style='font-family:"Space Mono",monospace; font-size:1rem; color:#00c8ff; font-weight:700; letter-spacing:0.05em;'>
            RAG · QUANTUM AI
        </div>
        <div style='font-size:0.7rem; color:#3a5870; margin-top:0.3rem; font-family:"Space Mono",monospace;'>
            arXiv:2505.23860v3
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='val'>{st.session_state.total_queries}</div>
            <div class='lbl'>Queries</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='val'>{len(st.session_state.messages)}</div>
            <div class='lbl'>Messages</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Settings
    st.markdown("#### ⚙️ Settings")
    top_k = st.slider("Chunks retrieved (top-k)", 1, 10, 5,
                      help="Number of relevant document chunks to retrieve per query.")
    show_sources = st.toggle("Show source chunks", value=True)

    st.divider()

    # Suggested questions
    st.markdown("#### 💡 Example Questions")
    example_qs = [
        "What is quantum machine learning?",
        "Quels sont les avantages du QRL?",
        "How does AI help in error correction?",
        "ما هي تطبيقات الذكاء الاصطناعي الكمومي؟",
        "What are the short-term goals of Quantum AI?",
    ]
    for eq in example_qs:
        if st.button(eq, use_container_width=True):
            st.session_state["pending_question"] = eq

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()

    st.markdown("""
    <div style='text-align:center; margin-top:1rem; font-size:0.68rem; color:#2a4060; font-family:"Space Mono",monospace;'>
        Powered by BAAI/bge-m3 · Qdrant · Llama 3.3 70B
    </div>""", unsafe_allow_html=True)


# ─── Main ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='quantum-header'>
    <span class='atom-icon'>⚛️</span>
    <h1>Quantum AI Assistant</h1>
    <div class='subtitle'>RAG · White Paper Intelligence · TP2</div>
</div>
""", unsafe_allow_html=True)

# Check credentials
if not all([GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY]):
    st.error("⚠️  Missing credentials. Add GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY to Streamlit secrets or environment variables.")
    st.stop()

# Load models with spinner
with st.spinner("🔄 Initializing embedder (first load: ~60s)…"):
    try:
        embedder = load_embedder()
        qdrant   = get_qdrant()
    except Exception as e:
        st.error(f"Connection error: {e}")
        st.stop()

# Welcome message
if not st.session_state.messages:
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, rgba(0,200,255,0.06), rgba(123,92,250,0.06));
        border: 1px solid rgba(0,200,255,0.15);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1rem 0 2rem;
        font-size: 0.92rem;
        line-height: 1.8;
        color: #8aa8c8;
    '>
        👋 Welcome! I'm your <strong style='color:#00c8ff'>Quantum AI RAG Assistant</strong>.<br>
        I answer questions based on the scientific white paper <strong>arXiv:2505.23860v3</strong> 
        on the intersection of AI and Quantum Computing.<br><br>
        Ask in <strong>English 🇬🇧</strong>, <strong>French 🇫🇷</strong>, or <strong>Arabic 🇸🇦</strong>.
    </div>
    """, unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "⚛️"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and "sources" in msg:
            with st.expander(f"📚 Source chunks ({len(msg['sources'])})", expanded=False):
                for s in msg["sources"]:
                    score_color = "#00c8ff" if s["score"] > 0.7 else "#7b5cfa" if s["score"] > 0.5 else "#4a6880"
                    st.markdown(f"""
                    <div style='border-left: 3px solid {score_color}; padding: 0.6rem 1rem; margin: 0.4rem 0;
                                background: rgba(255,255,255,0.02); border-radius: 0 8px 8px 0;'>
                        <span class='source-chip'>Chunk {s["chunk_id"]}</span>
                        <span style='font-family:"Space Mono",monospace; font-size:0.7rem; color:{score_color}; margin-left:0.5rem;'>
                            score: {s["score"]}
                        </span>
                        <div style='font-size:0.82rem; color:#6a8aaa; margin-top:0.4rem; font-style:italic;'>
                            {s["text"]}…
                        </div>
                    </div>""", unsafe_allow_html=True)

# Handle pending question from sidebar buttons
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")
else:
    question = st.chat_input("Ask anything about Quantum AI… (EN / FR / AR)")

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    # Build history for context
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    # Get answer
    with st.chat_message("assistant", avatar="⚛️"):
        with st.spinner("🔍 Searching quantum knowledge base…"):
            answer, sources = rag_query(question, top_k=top_k, history=history)
        st.markdown(answer)
        if show_sources and sources:
            with st.expander(f"📚 Source chunks ({len(sources)})", expanded=False):
                for s in sources:
                    score_color = "#00c8ff" if s["score"] > 0.7 else "#7b5cfa" if s["score"] > 0.5 else "#4a6880"
                    st.markdown(f"""
                    <div style='border-left: 3px solid {score_color}; padding: 0.6rem 1rem; margin: 0.4rem 0;
                                background: rgba(255,255,255,0.02); border-radius: 0 8px 8px 0;'>
                        <span class='source-chip'>Chunk {s["chunk_id"]}</span>
                        <span style='font-family:"Space Mono",monospace; font-size:0.7rem; color:{score_color}; margin-left:0.5rem;'>
                            score: {s["score"]}
                        </span>
                        <div style='font-size:0.82rem; color:#6a8aaa; margin-top:0.4rem; font-style:italic;'>
                            {s["text"]}…
                        </div>
                    </div>""", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.session_state.total_queries += 1
    st.rerun()
