# 🧠 Developer Diary — 27 October 2025

### **Focus:** Improving chatbot performance and RAG response accuracy

---

## 🎯 **Summary of Today’s Work**
Today’s session focused on improving the **efficiency, reliability, and response quality** of the Analytics Copilot chatbot.  
I continued refining the Retrieval-Augmented Generation (RAG) pipeline and dealt with several technical issues related to model loading, caching, and Hugging Face deployment.

---

## ⚙️ **Key Updates and Progress**

### **1. Model adjustments and hardware upgrade**
- I switched the chatbot from the **Mistral 7B Instruct** model to a **T4 Small GPU instance** on Hugging Face Spaces to improve response time.
- The model now loads faster and generates responses more smoothly.
- I learned that loading large LLMs (such as Llama 3 or Mistral) on CPU causes significant startup delays due to the model’s size (7B–8B parameters).  
  Upgrading to GPU significantly reduces latency.
- I also understood that each restart reloads the model weights and embeddings, causing slow start-ups unless caching is handled properly.

---

### **2. Debugging and fixing performance issues**
- I encountered several errors and learned how to fix them:
  - **`libgomp: Invalid value for OMP_NUM_THREADS`** → fixed by setting `os.environ["OMP_NUM_THREADS"]="1"`.
  - **`accelerate` device conflict** → resolved by removing `device=device` when using `device_map="auto"`.
  - **Missing `bs4` module** → installed `beautifulsoup4` to parse Confluence pages.
- These fixes made the script more stable and deployment-ready for both CPU and GPU environments.

---

### **3. Caching and response issues**
- Even after switching to T4 GPU, I noticed that **responses were cached**, causing the chatbot to return **outdated or irrelevant answers**.
- This revealed that cached embeddings or old results were being reused.
- I decided that the next improvement should be:
  > “The chatbot should only return the **relevant document link**, instead of dumping the entire context.”

This will make responses clearer, more focused, and user-friendly — especially for analytics documentation.

---

### **4. Understanding the tools better**
- I gained a deeper understanding of what each tool does:
  - **Gradio**: not an AI model, but a front-end framework to create the chat interface.
  - **SentenceTransformer**: used for embeddings and semantic search.
  - **LLM (Llama 3 or Mistral)**: generates natural language answers.
  - **Confluence & GitHub fetchers**: serve as document sources for RAG.

---

### **5. Free vs Paid Hugging Face setup**
- I learned how Hugging Face’s **free tier** works:
  - It provides **CPU-only** Spaces with limited compute and public visibility.
  - GPU options like **T4 Small** require an upgrade (Pro or paid).
- I now understand that the free tier disables outbound internet, meaning API-based GitHub or Confluence integrations only work on Pro plans.
- The current setup runs in **T4 Small GPU mode**, using cached embeddings for speed.

---

## 🚧 **Next Steps**
- Update the chatbot logic to:
  - Return **only the relevant document links**, not entire text snippets.
  - Possibly rank multiple sources and return the top few references.
- Improve caching behaviour so the chatbot refreshes embeddings intelligently rather than reusing stale data.
- Add a short **“fetch progress” indicator** for better UX.
- Consider integrating **Claude API fallback** later for enhanced reasoning and speed.
- Optionally, persist cache files across Space restarts to reduce cold-start times.

---

## 💡 **Key Learnings**
- Hugging Face Spaces (Free vs T4 GPU) and their limitations.  
- How `accelerate`, `device_map`, and threading interact in Transformers pipelines.  
- Caching and embeddings directly affect both performance and accuracy in RAG systems.  
- Good prompting and response truncation can drastically improve clarity.  
- Gradio is a UI framework — not a model — and must use the new `messages` format in v4.44+.

---

## 🗓️ **Reflection**
Today’s session deepened my understanding of **RAG architecture performance** and the **infrastructure side of model deployment**.  
I made significant progress in stabilising the chatbot, reducing startup errors, and planning improvements for accuracy and output formatting.  
The next phase will focus on making answers shorter, more relevant, and linked directly to their document sources.

---

### ✅ **End of Day Summary**
> “Moved the chatbot to GPU for faster inference, debugged major environment errors, fixed caching, and planned the next enhancement — to return only relevant document links for improved clarity.”
