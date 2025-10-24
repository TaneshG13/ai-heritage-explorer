
🏛️ AI Heritage Explorer
========================

**A conversational RAG assistant for art and history, powered by Llama 3.1 70b and Groq.**

<-- https://ai-heritage-explorer.streamlit.app/

_This application allows users to ask complex questions about cultural heritage. It retrieves factual context from Wikipedia and The Metropolitan Museum of Art, then uses a Large Language Model to synthesize that information into a detailed, conversational answer._

🌟 Core Features
----------------

*   **Conversational Memory:** Remembers the context of your chat, allowing for natural follow-up questions.
    
*   **Hybrid Data Retrieval (RAG):** Fetches information from two distinct sources:
    
    *   **Unstructured Text:** Full Wikipedia articles are retrieved and chunked for deep context.
        
    *   **Structured Data:** The Met Museum's API is queried for specific artifact details, images, and links.
        
*   **Real-Time Vector Search:** Dynamically creates a FAISS vector index from the retrieved documents for each new topic.
    
*   **High-Speed Generation:** Uses the **Llama 3.1 70b** model via the **Groq API** for incredibly fast and high-quality text generation.
    
*   **Cited Sources:** Every response is backed by the data it was generated from, with links to the original articles and artifacts.
    

🛠️ Tech Stack
--------------

This project combines several modern AI and web technologies:

*   **Frontend:** [Streamlit](https://streamlit.io/)
    
*   **Backend & Orchestration:** [Python](https://www.python.org/), [LangChain](https://www.langchain.com/)
    
*   **LLM:** [Llama 3.1 8b](https://llama.meta.com/) (via [Groq API](https://groq.com/))
    
*   **Retrieval:** [Wikipedia API](https://pypi.org/project/wikipedia/), [Met Museum API](https://metmuseum.github.io/)
    
*   **Vector Store:** [FAISS](https://faiss.ai/) (in-memory)
    
*   **Embeddings:** all-MiniLM-L6-v2 (via [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2))
    
*   **Deployment:** [Streamlit Community Cloud](https://streamlit.io/cloud)
    

🔄 How It Works: The RAG Pipeline
---------------------------------

1.  **User Query:** The user asks a question (e.g., "What was special about Vermeer's paintings?").
    
2.  **Contextualize:** The app checks the chat history. If it's a follow-up, it rephrases the query (e.g., "What was special about _Johannes Vermeer's_ paintings?").
    
3.  **Retrieve:**
    
    *   Fetches the full Wikipedia article for "Johannes Vermeer".
        
    *   Queries the Met Museum API for "Vermeer" and gets artifact data.
        
4.  **Chunk & Embed:** The Wikipedia article is split into small chunks. All retrieved documents are converted into vectors (embeddings) and loaded into a FAISS vector store.
    
5.  **Search:** The app searches this vector store for the chunks most relevant to the user's query.
    
6.  **Augment:** The most relevant chunks (the "context") are combined with the user's query into a detailed prompt.
    
7.  **Generate:** This prompt is sent to the Llama 3.1 70b model, which synthesizes a single, comprehensive answer.
    
8.  **Stream:** The answer is streamed back to the user's screen, and the retrieved sources are displayed.
    

🚀 Getting Started (Running Locally)
------------------------------------

You can run this project on your own machine by following these steps:

1.  git clone \[https://github.com/TaneshG13/ai-heritage-explorer.git\](https://github.com/TaneshG13/ai-heritage-explorer.git)cd ai-heritage-explorer
    
2.  \# On Windowspython -m venv venv.\\venv\\Scripts\\activate# On macOS/Linuxpython3 -m venv venvsource venv/bin/activate
    
3.  pip install -r requirements.txt
    
4.  **Set Up Your API Key:**
    
    *   Get a free API key from [Groq](https://groq.com/).
        
    *   Create a file at this _exact_ path: .streamlit/secrets.toml
        
    *   GROQ\_API\_KEY = "gsk\_YourKeyGoesHere"
        
5.  streamlit run app.pyYour browser will automatically open to the app.
    

☁️ Deployment
-------------

This application is deployed and hosted live on **Streamlit Community Cloud**. The deployment process is configured to:

*   Install all dependencies from requirements.txt.
    
*   Securely load the GROQ\_API\_KEY from the app's 'Secrets' settings.
