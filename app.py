import streamlit as st
import os
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Quantum AI — RAG Assistant",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Palette ──
   #C9788A  Puce
   #E29C9C  Salmon pink
   #E6BEAE  Pale Dogwood
   #EEDAC5  Almond
   #F5F5DC  Beige
*/

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F5F5DC !important;
    color: #3a2a2a !important;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(201,120,138,0.12) 0%, transparent 55%),
        radial-gradient(ellipse at 90% 100%, rgba(230,190,174,0.18) 0%, transparent 55%),
        #F5F5DC !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #EED8C8 0%, #E6BEAE 100%) !important;
    border-right: 1px solid rgba(201,120,138,0.25) !important;
}
[data-testid="stSidebar"] * { color: #5a3040 !important; }

/* ── Header ── */
.quantum-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.quantum-header h1 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3rem;
    font-weight: 600;
    font-style: italic;
    background: linear-gradient(135deg, #C9788A 0%, #a05060 60%, #C9788A 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: 0.01em;
    line-height: 1.1;
}
.quantum-header .subtitle {
    font-size: 0.78rem;
    color: #b08090;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-top: 0.5rem;
    font-weight: 500;
}
.atom-icon {
    font-size: 2.8rem;
    display: block;
    margin-bottom: 0.4rem;
    filter: drop-shadow(0 2px 8px rgba(201,120,138,0.35));
}

/* ── Chat container ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
}

/* ── Input ── */
[data-testid="stChatInputContainer"] {
    background: rgba(238,218,197,0.7) !important;
    border-top: 1px solid rgba(201,120,138,0.2) !important;
    backdrop-filter: blur(8px);
}
[data-testid="stChatInputTextArea"] {
    background: rgba(255,255,255,0.7) !important;
    border: 1.5px solid rgba(201,120,138,0.3) !important;
    border-radius: 14px !important;
    color: #3a2a2a !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInputTextArea"]:focus {
    border-color: #C9788A !important;
    box-shadow: 0 0 0 3px rgba(201,120,138,0.12) !important;
}

/* ── Message bubbles ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:last-child {
    background: linear-gradient(135deg, rgba(201,120,138,0.18), rgba(226,156,156,0.15)) !important;
    border: 1px solid rgba(201,120,138,0.25) !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 1rem 1.3rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div:last-child {
    background: rgba(255,255,255,0.65) !important;
    border: 1px solid rgba(230,190,174,0.5) !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 1rem 1.3rem !important;
    backdrop-filter: blur(6px);
}

/* ── Source chips ── */
.source-chip {
    display: inline-block;
    background: rgba(201,120,138,0.15);
    border: 1px solid rgba(201,120,138,0.35);
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: #a05060;
    margin: 0.15rem;
    letter-spacing: 0.05em;
}

/* ── Metric cards ── */
.metric-card {
    background: rgba(255,255,255,0.5);
    border: 1px solid rgba(201,120,138,0.2);
    border-radius: 14px;
    padding: 0.9rem;
    text-align: center;
    margin: 0.3rem 0;
    backdrop-filter: blur(4px);
}
.metric-card .val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.8rem;
    font-weight: 600;
    color: #C9788A;
    line-height: 1;
}
.metric-card .lbl {
    font-size: 0.68rem;
    color: #b09098;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.25rem;
}

/* ── Divider ── */
hr { border-color: rgba(201,120,138,0.18) !important; }

/* ── Buttons ── */
.stButton button {
    background: rgba(255,255,255,0.6) !important;
    border: 1.5px solid rgba(201,120,138,0.3) !important;
    color: #a05060 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    background: rgba(201,120,138,0.12) !important;
    border-color: #C9788A !important;
    box-shadow: 0 2px 12px rgba(201,120,138,0.2) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.45) !important;
    border: 1px solid rgba(230,190,174,0.5) !important;
    border-radius: 12px !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #C9788A !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(201,120,138,0.3); border-radius: 3px; }

/* ── Toggle ── */
[data-testid="stToggle"] input:checked + div { background: #C9788A !important; }

/* Hide branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def get_secret(key, default=""):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

GROQ_API_KEY    = get_secret("GROQ_API_KEY")
QDRANT_URL      = get_secret("QDRANT_URL")
QDRANT_API_KEY  = get_secret("QDRANT_API_KEY")
COLLECTION_NAME = get_secret("COLLECTION_NAME", "gots_chunks")


@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("BAAI/bge-m3")

@st.cache_resource(show_spinner=False)
def get_qdrant():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

@st.cache_resource(show_spinner=False)
def get_groq():
    return Groq(api_key=GROQ_API_KEY)


def rag_query(question, top_k=5, history=None):
    embedder = load_embedder()
    qdrant   = get_qdrant()
    groq     = get_groq()

    q_emb   = embedder.encode(question).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=q_emb,
        limit=top_k
    ).points

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
- If the answer is not in the context, reply: "Je n'ai pas trouvé cette information dans la base de connaissances."
- Cite chunk numbers [Chunk X] when referencing a source
- Distinguish timelines: short-term (3-5 yrs) / mid-term (5-10 yrs) / long-term (>10 yrs) when relevant
- Structure: Summary → Details → Source(s)
- Reply in the language of the question (French, English, or Arabic)""",
        }
    ]

    if history:
        for h in history[-6:]:
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


if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1.2rem 0 0.5rem;'>
        <div style='font-size:2.2rem'>⚛️</div>
        <div style='font-family:"Cormorant Garamond",serif; font-size:1.15rem;
                    color:#C9788A; font-weight:600; font-style:italic; letter-spacing:0.04em;'>
            RAG · Quantum AI
        </div>
        <div style='font-size:0.68rem; color:#c0a0a8; margin-top:0.25rem; letter-spacing:0.1em;'>
            arXiv:2505.23860v3
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='val'>{st.session_state.total_queries}</div>
            <div class='lbl'>Queries</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='val'>{len(st.session_state.messages)}</div>
            <div class='lbl'>Messages</div></div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### ⚙️ Settings")
    top_k = st.slider("Chunks retrieved (top-k)", 1, 10, 5)
    show_sources = st.toggle("Show source chunks", value=True)

    st.divider()
    st.markdown("#### 💡 Example Questions")
    example_qs = [
        "What is quantum machine learning?",
        "Quels sont les avantages du QRL?",
        "How does AI help in error correction?",
        "ما هي تطبيقات الذكاء الاصطناعي الكمومي؟",
        "What are short-term goals of Quantum AI?",
    ]
    for eq in example_qs:
        if st.button(eq, use_container_width=True):
            st.session_state["pending_question"] = eq

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()

    st.markdown("""
    <div style='text-align:center;margin-top:1rem;font-size:0.68rem;color:#c0a8b0;'>
        BAAI/bge-m3 · Qdrant · Llama 3.3 70B
    </div>""", unsafe_allow_html=True)


# ── Main ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='quantum-header'>
    <span class='atom-icon'>⚛️</span>
    <h1>Quantum AI Assistant</h1>
    <div class='subtitle'>RAG · White Paper Intelligence</div>
</div>
""", unsafe_allow_html=True)

if not all([GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY]):
    st.error("⚠️ Missing credentials. Add GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY to Streamlit secrets.")
    st.stop()

with st.spinner("Initializing… (first load: ~60s)"):
    try:
        load_embedder()
        get_qdrant()
    except Exception as e:
        st.error(f"Connection error: {e}")
        st.stop()

if not st.session_state.messages:
    st.markdown("""
    <div style='
        background: rgba(255,255,255,0.55);
        border: 1px solid rgba(201,120,138,0.2);
        border-radius: 18px;
        padding: 1.6rem 2rem;
        margin: 0.5rem 0 2rem;
        font-size: 0.93rem;
        line-height: 1.85;
        color: #7a5060;
        backdrop-filter: blur(6px);
    '>
        👋 Welcome! I'm your <strong style="color:#C9788A">Quantum AI RAG Assistant</strong>.<br>
        I answer questions based on the scientific white paper <strong>arXiv:2505.23860v3</strong>
        on the intersection of AI and Quantum Computing.<br><br>
        Ask in <strong>English 🇬🇧</strong>, <strong>French 🇫🇷</strong>, or <strong>Arabic 🇸🇦</strong>.
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "⚛️"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and "sources" in msg:
            with st.expander(f"📚 Source chunks ({len(msg['sources'])})", expanded=False):
                for s in msg["sources"]:
                    score_color = "#C9788A" if s["score"] > 0.7 else "#E29C9C" if s["score"] > 0.5 else "#b0a0a8"
                    st.markdown(f"""
                    <div style='border-left:3px solid {score_color}; padding:0.6rem 1rem; margin:0.4rem 0;
                                background:rgba(255,255,255,0.5); border-radius:0 10px 10px 0;'>
                        <span class='source-chip'>Chunk {s["chunk_id"]}</span>
                        <span style='font-size:0.72rem; color:{score_color}; margin-left:0.4rem; font-weight:600;'>
                            score: {s["score"]}
                        </span>
                        <div style='font-size:0.82rem; color:#9a7880; margin-top:0.35rem; font-style:italic;'>
                            {s["text"]}…
                        </div>
                    </div>""", unsafe_allow_html=True)

if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")
else:
    question = st.chat_input("Ask anything about Quantum AI… (EN / FR / AR)")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]

    with st.chat_message("assistant", avatar="⚛️"):
        with st.spinner("Searching knowledge base…"):
            answer, sources = rag_query(question, top_k=top_k, history=history)
        st.markdown(answer)
        if show_sources and sources:
            with st.expander(f"📚 Source chunks ({len(sources)})", expanded=False):
                for s in sources:
                    score_color = "#C9788A" if s["score"] > 0.7 else "#E29C9C" if s["score"] > 0.5 else "#b0a0a8"
                    st.markdown(f"""
                    <div style='border-left:3px solid {score_color}; padding:0.6rem 1rem; margin:0.4rem 0;
                                background:rgba(255,255,255,0.5); border-radius:0 10px 10px 0;'>
                        <span class='source-chip'>Chunk {s["chunk_id"]}</span>
                        <span style='font-size:0.72rem; color:{score_color}; margin-left:0.4rem; font-weight:600;'>
                            score: {s["score"]}
                        </span>
                        <div style='font-size:0.82rem; color:#9a7880; margin-top:0.35rem; font-style:italic;'>
                            {s["text"]}…
                        </div>
                    </div>""", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.session_state.total_queries += 1
    st.rerun()
