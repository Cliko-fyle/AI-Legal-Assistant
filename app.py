import json
import torch
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

#Loading Datasets
with open('ipc_qa.json') as f1, open('crpc_qa.json') as f2:
    ipc_data = json.load(f1)
    crpc_data = json.load(f2)

documents = ipc_data + crpc_data
corpus = [doc['question'] + " " + doc['answer'] for doc in documents]

#Embedding Model
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=False)

#FAISS Indexing
dimension = len(corpus_embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(corpus_embeddings))

# Q&A Model function
@st.cache_resource
def load_qa_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline(
        "text2text-generation",
        model = model,
        tokenizer = tokenizer,
        device = 0 if torch.cuda.is_available() else -1
    )

qa_model = load_qa_model()

#Function for retrieval
def retrieve(query, k=3):
    query_vec = embedder.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    return [corpus[i] for i in indices[0]]

#Function for answer generation
def generate_answer(query):
    retrieved_docs = retrieve(query)
    context = "\n".join(retrieved_docs)
    prompt = f"Answer the legal question based on the context below:\n{context}\n\nQuestion: {query}"
    response = qa_model(prompt, max_new_tokens=200)[0]['generated_text']
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



