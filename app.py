import streamlit as st
from dotenv import load_dotenv
import os

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

HF_TOKEN = os.getenv("HUGGING_FACE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot (HF + Pinecone)")

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-base-en-v1.5")

embed_model = load_embedding_model()

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x22B-Instruct-v0.1",
    huggingfacehub_api_token=HF_TOKEN,
    task="text-generation",
    temperature=0,
    max_new_tokens=1000
)

chat_model = ChatHuggingFace(llm=llm)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask something from your knowledge base...")

if user_query:

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    query_vector = embed_model.encode(user_query).tolist()

    response = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )

    contexts = [
        match["metadata"].get("text", "")
        for match in response["matches"]
    ]

    context = "\n\n".join(contexts)

    system_prompt = f"""
You are a helpful assistant.
 -If user say greeting to you or his/her name then store in your context and respond to the user 
 accordingly

 -if user ask the system on difference between some topics then return answer in tabular format
Answer ONLY using the context below..

Context:
{context}
"""

    lc_messages = [SystemMessage(content=system_prompt)]

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))

    llm_response = chat_model.invoke(lc_messages)
    answer = llm_response.content

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)
