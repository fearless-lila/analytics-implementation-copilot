# =============================================
#  Analytics Copilot (RAG + Llama 3 8B 4-bit)
#  Optimized for NVIDIA T4 GPUs with caching
# =============================================

import os
import base64
import json
import torch
import requests
import gradio as gr
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ---------------------------------------------
# 🔐 1. Load tokens
# ---------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")           # Hugging Face access token
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")   # GitHub token (optional for public repos)

# ---------------------------------------------
# ⚙️ 2. Initialize embedding model
# ---------------------------------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

device = 0 if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {'GPU' if device == 0 else 'CPU'}")

# ---------------------------------------------
# 🧠 3. Load chat model (Llama 3 8B 4-bit)
# ---------------------------------------------
model_name = "meta-llama/Meta-Llama-3-8B-Instruct-4bit"
print(f"🚀 Using {model_name}")

chat_model = pipeline(
    "text-generation",
    model=model_name,
    token=HF_TOKEN,
    device=device,
    dtype="auto"
)

# ---------------------------------------------
# 📂 4. Fetch files from GitHub
# ---------------------------------------------
def fetch_github_files(repo="fearless-lila/analytics-implementation-copilot", token=None):
    """Fetch text-based files from a GitHub repository."""
    headers = {"Authorization": f"token {token}"} if token else {}
    url = f"https://api.github.com/repos/{repo}/git/trees/main?recursive=1"
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        print("❌ Error fetching file list:", r.text)
        return []

    files = []
    for item in r.json().get("tree", []):
        if item["type"] != "blob":
            continue
        path = item["path"]
        # Filter text-like files
        if any(path.lower().endswith(ext) for ext in [".py", ".md", ".txt", ".yaml", ".yml", ".json"]):
            content_url = f"https://api.github.com/repos/{repo}/contents/{path}"
            resp = requests.get(content_url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                if "content" in data:
                    content = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
                    files.append({
                        "path": path,
                        "url": f"https://github.com/{repo}/blob/main/{path}",
                        "content": content
                    })
    print(f"✅ Loaded {len(files)} files from {repo}")
    return files

# ---------------------------------------------
# 💾 5. Cache embeddings
# ---------------------------------------------
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
EMB_PATH = os.path.join(CACHE_DIR, "embeddings.pt")
FILES_PATH = os.path.join(CACHE_DIR, "files.json")

def prepare_embeddings(files):
    """Generate or load cached embeddings for GitHub files."""
    if os.path.exists(EMB_PATH) and os.path.exists(FILES_PATH):
        print("💾 Loading cached embeddings...")
        embeddings = torch.load(EMB_PATH)
        with open(FILES_PATH, "r") as f:
            cached_files = json.load(f)
        if len(cached_files) == len(files):
            print("✅ Using cached embeddings.")
            return cached_files, embeddings
        else:
            print("⚠️ Repo changed — regenerating embeddings.")

    print("🧮 Generating new embeddings...")
    valid_files = [f for f in files if f["content"].strip()]
    texts = [f["content"] for f in valid_files]
    embeddings = embedder.encode(texts, convert_to_tensor=True)

    torch.save(embeddings, EMB_PATH)
    with open(FILES_PATH, "w") as f:
        json.dump(valid_files, f)

    print(f"✅ Cached {len(valid_files)} embeddings.")
    return valid_files, embeddings

# ---------------------------------------------
# 🔍 6. Retrieve top docs for RAG
# ---------------------------------------------
def retrieve_top_docs(query, files, embeddings, top_k=2):
    """Return top-K relevant file snippets for the query."""
    q_emb = embedder.encode(query, convert_to_tensor=True)
    sims = util.cos_sim(q_emb, embeddings)[0]
    best = sims.topk(top_k)
    docs = []
    for score, idx in zip(best.values, best.indices):
        f = files[idx]
        snippet = f["content"][:300].replace("\n", " ")
        docs.append(f"File: {f['path']} — {f['url']}\n{snippet}...")
    return "\n\n".join(docs)

# ---------------------------------------------
# 💬 7. RAG-powered chat
# ---------------------------------------------
def rag_answer(user_msg, history, files, embeddings):
    context = retrieve_top_docs(user_msg, files, embeddings, top_k=2)
    prompt = f"""
You are an analytics assistant. Use the following project files to answer the user's question.
Be concise, factual, and avoid guessing.

Context:
{context}

Conversation:
{history[-5:]}

User: {user_msg}
AI:"""

    out = chat_model(
        prompt,
        max_new_tokens=150,
        temperature=0.4,
        do_sample=True
    )[0]["generated_text"]
    reply = out.split("AI:")[-1].strip()
    return reply

# ---------------------------------------------
# 🗨️ 8. Gradio Chat Interface
# ---------------------------------------------
def chat_fn(message, history):
    if not files or embeddings is None or len(embeddings) == 0:
        return "⚠️ No documents loaded from GitHub."
    reply = rag_answer(message, history, files, embeddings)
    history.append((message, reply))
    return history, history

# ---------------------------------------------
# 🚀 9. Load repo, build cache & launch app
# ---------------------------------------------
repo_name = "fearless-lila/analytics-implementation-copilot"

files = fetch_github_files(repo=repo_name, token=GITHUB_TOKEN)
files, embeddings = prepare_embeddings(files)
print(f"✅ Ready with {len(files)} files indexed for RAG.")

gr.ChatInterface(
    fn=chat_fn,
    title="Analytics Copilot — Llama 3 8B 4-bit RAG",
    description="Ask questions about analytics specs or files in your GitHub repo.",
    theme="soft",
    type="messages"
).launch()