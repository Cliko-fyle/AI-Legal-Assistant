import json
import torch
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

#Loading Datasets
with open('ipc_qa.json') as f1, open('crpc_qa.json') as f2:
    ipc_data = json.load(f1)
    crpc_data = json.load(f2)

documents = ipc_data + crpc_data
corpus = [doc['question'] + " " + doc['answer'] for doc in documents]


#FAISS Indexing
@st.cache_resource
def faiss_indexing(corpus):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    corpus_embeddings = embedder.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
    index.add(corpus_embeddings)
    return embedder, index

embedder, index = faiss_indexing(corpus)

# Q&A Model function
@st.cache_resource
def load_qa_model():
    model_name = "EleutherAI/gpt-neo-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return pipeline(
        "text-generation",
        model = model,
        tokenizer = tokenizer,
    )

qa_model = load_qa_model()

#Function for retrieval
def retrieve(query, k=3):
    query_vec = embedder.encode([query], convert_to_numpy = True, normalize_embeddings = True)
    distances, indices = index.search(query_vec, k)
    return [corpus[i] for i in indices[0]]

#Function for answer generation
def generate_answer(query):
    retrieved_docs = retrieve(query)
    context = "\n".join(retrieved_docs)
    prompt = (
        f"You are a knowledgeable legal assistant. "
        f"Answer the question in a clear and informative manner, using only the context provided. "
        f"Explain briefly with reasoning or details when possible, but stay relevant. "
        f"If the answer is not in the context, say 'The information is not available in the provided documents.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    response = qa_model(prompt, max_new_tokens=200, temperature=0.3)[0]['generated_text']
    return response.strip()

#App deployment using STREAMLIT
st.set_page_config(page_title="⚖️ AI-Powered Legal Assistant", layout="centered")
st.title("⚖️ AI-Powered Legal Assistant")
st.write("Ask any legal question and get an AI-generated answer")

query = st.text_area("Enter your legal question:", height=100, placeholder="E.g. What rights does an arrested person have?")
if st.button("Get Answer"):
    if query.strip():
        with st.spinner("Generating answer..."):
            answer = generate_answer(query)
        st.subheader("Answer:")
        st.write(answer)
    else:

        st.warning("Please enter a valid question before submitting.")
















