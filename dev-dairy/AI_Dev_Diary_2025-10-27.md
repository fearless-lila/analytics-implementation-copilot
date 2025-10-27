# 🧠 AI Learning Diary — 27 October 2025

## 🗓️ Focus of the Day
Enhancing and optimising my Hugging Face RAG chatbot — from model selection and debugging to performance tuning and deployment.

---

## 🧩 What I Worked On

Today, I concentrated on improving my **Analytics Copilot chatbot**, which uses **Retrieval-Augmented Generation (RAG)** to answer questions from my GitHub repository.

1. **Revisiting the chatbot code**  
   - The previous version used **BLOOMZ-560M**, but it generated inconsistent and irrelevant responses.  
   - I realised this was because BLOOMZ-560M isn’t instruction-tuned, so it struggles with multi-turn conversations.

2. **Model exploration and comparison**  
   - Explored alternatives such as **Mistral 7B Instruct** and **Meta Llama 3 8B Instruct**.  
   - Learned that *instruction-tuned* models are designed to follow user prompts better, provide concise answers, and maintain conversational context.  
   - Decided to begin testing **Mistral 7B**, as it is openly available and performs well for chat-based tasks.

3. **Switching to Llama 3**  
   - Eventually upgraded to **Meta Llama 3 8B Instruct (4-bit)** for faster and more efficient inference.  
   - Updated the chatbot to load Llama 3 automatically and optimised its configuration for the hardware used.

4. **Debugging and errors**  
   - Encountered and resolved multiple issues:  
     - `torch_dtype` deprecation warning → fixed by using `dtype="auto"`.  
     - Missing `accelerate` dependency → resolved by installing `accelerate`.  
     - Boolean tensor check (`RuntimeError: Boolean value of Tensor with more than one value is ambiguous`) → fixed by replacing `if not embeddings:` with `embeddings is None or len(embeddings) == 0`.  
     - Gradio warning about deprecated message format → fixed by switching to `type="messages"` in the chat interface.

5. **Performance optimisation**  
   - Llama 3 produced excellent responses but initially ran slowly.  
   - Discovered that this was due to model size and CPU inference.  
   - Learned about hardware differences — **NVIDIA T4 (16 GB)** GPUs can efficiently run 7B–8B models using quantisation.  
   - Replaced the full-precision model with **`meta-llama/Meta-Llama-3-8B-Instruct-4bit`**, optimised for T4 GPUs.  
   - Reduced `max_new_tokens` and adjusted retrieval parameters for faster replies.

6. **Caching embeddings**  
   - Implemented a caching system to save embeddings in `/cache/embeddings.pt` and `/cache/files.json`.  
   - When the app restarts, it loads cached embeddings instead of regenerating them, drastically reducing startup time.

---

## 💡 Key Things I Learned

| Topic | What I Learned |
|-------|----------------|
| **Model tuning** | Instruction-tuned models like Mistral and Llama 3 yield far better conversational performance. |
| **Vector vs Embedding** | Vectors are numerical representations; embeddings are meaningful vector representations of text. |
| **Device and acceleration** | The `accelerate` library is required for `device_map="auto"` to distribute model weights efficiently. |
| **Performance trade-offs** | Llama 3 8B provides high-quality answers but is slower without GPU acceleration; the 4-bit version is much faster. |
| **Caching** | Pre-computed embeddings significantly reduce load time. |
| **Gradio improvements** | Using `type="messages"` aligns with the latest Gradio chat message standards. |

---

## ⚙️ Final Setup

- **Model:** `meta-llama/Meta-Llama-3-8B-Instruct-4bit`  
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`  
- **Hardware:** NVIDIA T4 GPU (16 GB)  
- **Frameworks:** Transformers, SentenceTransformers, Gradio  
- **Features:**  
  - RAG-based document retrieval  
  - GitHub repository integration  
  - Embedding cache  
  - Auto model selection (Llama 3 / Mistral fallback)

---

## 🎧 Learning Habit Update
Today, I also began **watching and listening to AI-related podcasts**, and found them an excellent way to learn.  
Podcasts help me absorb complex ideas, understand real-world applications, and improve my intuition through expert discussions.

---

## 🧭 Next Steps
- Add automatic update detection for cached embeddings (based on the latest GitHub commit).  
- Display a **startup summary** in the Gradio UI showing:  
  - Loaded model (Llama 3 / Mistral)  
  - Device in use (GPU or CPU)  
  - Number of indexed files  
- Begin experimenting with **document chunking** to improve retrieval accuracy for large files.
