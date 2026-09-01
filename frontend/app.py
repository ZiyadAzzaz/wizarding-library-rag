import httpx
import streamlit as st
from api_client import RagApiClient

st.set_page_config(page_title="Wizarding Library", page_icon="⚡", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background: radial-gradient(circle at top left,#30234d,#0d0b14 50%);}
    [data-testid="stChatMessage"] {
      border:1px solid #6c558a55;border-radius:16px;background:#171221cc;
    }
    .source {padding:.7rem;border-left:3px solid #d9b45f;background:#18131f;margin:.35rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

client = RagApiClient()
st.title("⚡ Wizarding Library")
st.caption("Grounded answers from the seven Harry Potter books, with page-level citations.")

with st.sidebar:
    st.header("Retrieval settings")
    top_k = st.slider("Sources", 1, 10, 4)
    if st.button("Check backend", use_container_width=True):
        try:
            health = client.health()
            if health["status"] == "ok":
                st.success(f"Ready · {health['collection']}")
            else:
                st.warning("Backend is running, but the vector store is unavailable.")
        except httpx.HTTPError as exc:
            st.error(f"Backend unavailable: {exc}")
    st.info(
        "Answers are limited to retrieved book excerpts. "
        "Missing evidence returns an honest 'I do not know'."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for source in message.get("sources", []):
            st.markdown(
                f"<div class='source'><b>{source['document']}</b> · page {source['page']} · "
                f"score {source['score']:.3f}<br><small>{source['excerpt']}</small></div>",
                unsafe_allow_html=True,
            )

question = st.chat_input("Ask about a character, place, object, or event…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the library and grounding the answer…"):
            try:
                result = client.query(question, top_k)
                st.markdown(result["answer"])
                for source in result["sources"]:
                    st.markdown(
                        f"<div class='source'><b>{source['document']}</b>"
                        f" · page {source['page']} · "
                        f"score {source['score']:.3f}<br><small>{source['excerpt']}</small></div>",
                        unsafe_allow_html=True,
                    )
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
                )
            except httpx.HTTPStatusError as exc:
                detail = exc.response.json().get("detail", "Request failed")
                st.error(f"The assistant is not ready: {detail}")
            except httpx.HTTPError:
                st.error("I could not reach the backend. Start FastAPI and check API_BASE_URL.")
