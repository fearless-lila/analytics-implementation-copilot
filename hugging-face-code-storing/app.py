import os
import requests
import gradio as gr
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# -------------------------------
# Load AI models
# -------------------------------
# Semantic search for documents
search_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# LLM for human-like chat and answering questions
chat_model = pipeline(
    "text-generation",
    model="bigscience/bloomz-560m",
    tokenizer="bigscience/bloomz-560m"
)

# -------------------------------
# Fetch GitHub files with URLs and content
# -------------------------------
def fetch_github_files(repo="fearless-lila/analytics-implementation-copilot", token=None):
    headers = {"Authorization": f"token {token}"} if token else {}
    url = f"https://api.github.com/repos/{repo}/git/trees/main?recursive=1"
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        print("Error fetching file list:", r.text)
        return []

    files = []
    for item in r.json().get("tree", []):
        if item["type"] == "blob":
            path = item["path"]
            github_url = f"https://github.com/{repo}/blob/main/{path}"

            # Fetch content
            raw_url = f"https://raw.githubusercontent.com/{repo}/main/{path}"
            content_r = requests.get(raw_url, headers=headers)
            content = content_r.text if content_r.status_code == 200 else ""
            files.append({"path": path, "url": github_url, "content": content})
    return files

# -------------------------------
# Prepare embeddings for GitHub files
# -------------------------------
def prepare_file_embeddings(files):
    paths = [f["path"] for f in files]
    contents = [f["content"] for f in files]
    embeddings = search_model.encode(contents, convert_to_tensor=True)
    return embeddings

# -------------------------------
# RAG search: retrieve top file contents based on query
# -------------------------------
def retrieve_top_docs(query, files, embeddings, top_k=3):
    query_emb = search_model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_emb, embeddings)[0]
    top_results = similarities.topk(top_k)
    top_docs = []
    for score, idx in zip(top_results.values, top_results.indices):
        f = files[idx]
        top_docs.append(f"File: {f['path']} — {f['url']}\n{f['content'][:500]}...")  # first 500 chars
    return top_docs

# -------------------------------
# Conversational AI response with RAG
# -------------------------------
conversation_history = []

def rag_conversational_response(message, files, embeddings):
    global conversation_history
    conversation_history.append(f"User: {message}")

    # Retrieve top docs
    top_docs = retrieve_top_docs(message, files, embeddings, top_k=3)
    docs_text = "\n\n".join(top_docs)

    # Build prompt
    prompt = "\n".join(conversation_history[-10:]) + "\nRelevant Docs:\n" + docs_text + "\nAI:"
    response = chat_model(prompt, max_length=300, do_sample=True, temperature=0.7)[0]["generated_text"]

    ai_reply = response[len(prompt):].strip()
    conversation_history.append(f"AI: {ai_reply}")
    return ai_reply

# -------------------------------
# Handle query
# -------------------------------
# Preload files and embeddings once at startup
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
files = fetch_github_files(token=GITHUB_TOKEN)
embeddings = prepare_file_embeddings(files)

def handle_query(message):
    # Simple heuristic: if it contains "document" keywords, do RAG
    doc_keywords = ["document", "repo", "file", "spec", "tracking", "analytics"]
    if any(word in message.lower() for word in doc_keywords):
        return rag_conversational_response(message, files, embeddings)
    else:
        # General conversation without RAG
        conversation_history.append(f"User: {message}")
        prompt = "\n".join(conversation_history[-10:]) + "\nAI:"
        response = chat_model(prompt, max_length=150, do_sample=True, temperature=0.7)[0]["generated_text"]
        ai_reply = response[len(prompt):].strip()
        conversation_history.append(f"AI: {ai_reply}")
        return ai_reply

# -------------------------------
# Gradio Chat Interface
# -------------------------------
def chat_fn(message, history):
    response = handle_query(message)
    return response

gr.ChatInterface(fn=chat_fn, title="Analytics Copilot (RAG + BLOOMZ Chat)").launch()