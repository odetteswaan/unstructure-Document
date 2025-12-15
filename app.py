import streamlit as st
from dotenv import load_dotenv
import os

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# -------------------------------
# Load Environment Variables
# -------------------------------
load_dotenv()

HF_TOKEN = os.getenv("HUGGING_FACE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

# -------------------------------
# Streamlit UI (UNCHANGED)
# -------------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot (HF + Pinecone)")

# -------------------------------
# Session State
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# Load Embedding Model
# -------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-base-en-v1.5")

embed_model = load_embedding_model()

# -------------------------------
# Initialize Pinecone
# -------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# -------------------------------
# Initialize HuggingFace LLM
# -------------------------------
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x22B-Instruct-v0.1",
    huggingfacehub_api_token=HF_TOKEN,
    task="text-generation",
    temperature=0,
    max_new_tokens=1000
)

chat_model = ChatHuggingFace(llm=llm)

# -------------------------------
# Display Chat History (UNCHANGED)
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# User Input
# -------------------------------
user_query = st.chat_input("Ask something from your knowledge base...")

if user_query:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # -------------------------------
    # Embed Query
    # -------------------------------
    query_vector = embed_model.encode(user_query).tolist()

    # -------------------------------
    # Query Pinecone
    # -------------------------------
    response = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )

    # -------------------------------
    # Build Context
    # -------------------------------
    contexts = [
        match["metadata"].get("text", "")
        for match in response["matches"]
    ]

    context = "\n\n".join(contexts)

    # -------------------------------
    # System Prompt
    # -------------------------------
    system_prompt = f"""
You are a helpful assistant.
Answer ONLY using the context below..

Context:
{context}
"""

    # -------------------------------
    # Build FULL conversation history for LLM
    # -------------------------------
    lc_messages = [SystemMessage(content=system_prompt)]

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))

    # -------------------------------
    # LLM Response
    # -------------------------------
    llm_response = chat_model.invoke(lc_messages)
    answer = llm_response.content

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)
