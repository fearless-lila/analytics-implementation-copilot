# 🧠 Analytics Copilot — Development Diary & Learning Notes

**Date:** October 26, 2025  
**Goal:** Build a free AI chatbot that can search documentation (GitHub, Confluence, Figma, etc.) and support natural conversation.

---

## 🚀 Development Progress

Secret key not working once push the change to github, so need to get manully from notes, not sure why.

### **1️⃣ Base Chatbot Setup**
- Created the first version using **Gradio** on Hugging Face.
- Connected to **GitHub API** to search for files by name or keyword.
- Learned to run code locally in VS Code and test GitHub API authentication.

**Key takeaways:**
- Hugging Face Spaces doesn’t run code locally — it hosts it online.  
- GitHub API requires a **personal access token** (PAT) for authenticated searches.  
- “Requires authentication” errors usually mean the token isn’t being passed correctly.

---

### **2️⃣ Semantic Search (Embeddings)**
- Added **`sentence-transformers/all-MiniLM-L6-v2`** to enable **semantic understanding** (meaning-based search).  
- Learned what **embeddings** are and why they’re powerful:
  - They represent the **meaning** of text as vectors.
  - Allow the model to find files that “mean the same thing,” even if the wording differs.
  - For example, “checkout tracking” ≈ “purchase flow analytics”.

**Key learnings:**
- Embeddings = numerical representation of text meaning.  
- Cosine similarity measures how close meanings are.  
- Without embeddings, the bot only does literal keyword search.  

---

### **3️⃣ Conversational AI (DialoGPT-small)**
- Added a **conversation model** so the bot can respond to greetings.
- Learned that `pipeline("conversational")` was **deprecated** in new `transformers`.
- Fixed it by using **`pipeline("text-generation")`** instead.

**Key learnings:**
- The “weird” conversations were because small models (like DialoGPT-small)  
  can’t reason deeply or maintain long context.  
- Chatbots that *talk back like humans* need **LLMs** (Large Language Models).

---

### **4️⃣ Understanding LLMs**
- Switched to **`EleutherAI/gpt-neo-2.7B`** — an open-source **LLM**.  
- Learned that:
  - It *is* an LLM (2.7 billion parameters).  
  - It’s smaller than GPT-3/4, so responses are weaker.  
  - LLMs generate text by predicting the next word based on context.  

**Key learnings:**
- “LLM” doesn’t always mean “GPT-4 level.”  
- More parameters = deeper reasoning, but slower and heavier.  
- Open models like GPT-Neo or BLOOMZ are **free**, but require more prompt engineering.

---

### **5️⃣ Upgrading to BLOOMZ-560M**
- Replaced GPT-Neo with **`bigscience/bloomz-560m`** for Hugging Face free-tier compatibility.
- BLOOMZ is instruction-tuned → better at following prompts.
- Learned that instruction-tuned = fine-tuned to respond conversationally to user input.

**Key learnings:**
- BLOOMZ is small, free, and great for basic chatbots.  
- Chat still “feels weird” because the model isn’t reasoning — it’s just generating.  
- Larger LLMs or retrieval-augmented prompts improve intelligence.

---

### **6️⃣ Adding RAG (Retrieval-Augmented Generation)**
- Implemented true **RAG pipeline**:
  - Fetch GitHub file contents via `raw.githubusercontent.com`.  
  - Embed all file contents with Sentence Transformers.  
  - Retrieve top relevant files for each user query.  
  - Feed file snippets + conversation history into BLOOMZ.  

**Now the chatbot:**
- Reads file contents.  
- Generates answers *based on actual documentation*.  
- Mixes conversation + retrieval = full **RAG behavior**.

**Key learnings:**
- RAG = Retrieve + Read + Generate.  
- The model itself doesn’t “learn” — it’s still frozen.  
- RAG improves relevance without fine-tuning the model.  

#### Note: this convo is still weird and not returning the spec link, need to continue tomorrow to find a solution.
---

### **7️⃣ Visualization**
- Created a **flowchart** showing chatbot logic:
  1. User sends a query.  
  2. Check if it’s document-related.  
  3. Retrieve relevant GitHub files via embeddings.  
  4. Feed docs + chat history into BLOOMZ.  
  5. Generate an answer.  

**Learned:**  
- A clear diagram helps document architecture for teammates or future reference.  

---

## 🧩 What I Learned Today

| Topic | Key Learning |
|-------|---------------|
| **Hugging Face Spaces** | You don’t run code locally; it’s hosted online. |
| **GitHub API** | Needs token authentication; 401 errors mean missing/invalid token. |
| **Embeddings** | They capture *meaning*, not just words — essential for semantic search. |
| **Semantic Search vs Keyword Search** | Semantic = meaning-based; Keyword = literal text match. |
| **LLMs (Large Language Models)** | Generate human-like text by predicting next words; bigger = smarter. |
| **DialoGPT vs LLMs** | DialoGPT is small and chat-only; LLMs can reason and adapt better. |
| **RAG Concept** | Combine retrieval + generation → model can answer using real data. |
| **Open-source Models** | BLOOMZ, GPT-Neo, MPT, Alpaca are free LLMs for experimentation. |
| **Transformers Pipeline Changes** | “conversational” task was deprecated — use “text-generation”. |
| **Token & Secrets** | Hugging Face Spaces stores secrets in environment variables (e.g., `GITHUB_TOKEN`). |






---

## ✅ Current Capabilities

- Human-like chat (using BLOOMZ-560M)  
- Semantic GitHub search (embeddings)  
- Full RAG: answers based on real file content  
- Free to deploy and run  

---

## 🔜 Next Steps

- Add **Confluence** and **Figma** document search.  
- Summarize large documents before passing to LLM.  
- Add caching or FAISS for faster retrieval.  
- (Future) Data anomaly detection for analytics tracking.

---

### 📝 README Reminder

> This chatbot is **partially RAG-based** — it retrieves and reads GitHub file content using open models.  
> It does **not learn or fine-tune itself** over time.  
> All models are **pretrained and frozen**; they generate responses based on prompt and retrieved data.  
