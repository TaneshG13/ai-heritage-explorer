import streamlit as st
import requests
import wikipedia
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- 1. Functions for Data Retrieval (The "R" in RAG) ---

@st.cache_data(show_spinner="Fetching Wikipedia data...")
def get_wikipedia_data(query: str, max_docs=3) -> list[Document]:
    """Fetches data from Wikipedia and returns a list of LangChain Documents."""
    try:
        results = wikipedia.search(query, results=max_docs)
        summaries = []
        for res in results:
            try:
                page = wikipedia.page(res, auto_suggest=False)
                summaries.append(Document(
                    page_content=page.summary,
                    metadata={"source": "wikipedia", "title": page.title, "url": page.url}
                ))
            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
                continue # Skip disambiguation or non-existent pages
        return summaries
    except Exception as e:
        st.error(f"Error fetching from Wikipedia: {e}")
        return []

@st.cache_data(show_spinner="Fetching Met Museum data...")
def get_met_museum_data(query: str, max_docs=3) -> list[Document]:
    """Fetches artifact data from The Met Museum and returns a list of LangChain Documents."""
    try:
        search_url = f"https://collectionapi.metmuseum.org/public/collection/v1/search?q={query}&hasImages=true"
        response = requests.get(search_url)
        response.raise_for_status()
        search_data = response.json()
        
        if not search_data.get('objectIDs'):
            return []
            
        object_ids = search_data['objectIDs'][:max_docs]
        artifacts = []
        for obj_id in object_ids:
            try:
                obj_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}"
                obj_response = requests.get(obj_url)
                obj_response.raise_for_status()
                obj_data = obj_response.json()
                
                content = f"Title: {obj_data.get('title', 'N/A')}\n" \
                          f"Artist: {obj_data.get('artistDisplayName', 'N/A')}\n" \
                          f"Date: {obj_data.get('objectDate', 'N/A')}\n" \
                          f"Medium: {obj_data.get('medium', 'N/A')}\n" \
                          f"Description: {obj_data.get('objectName', 'N/A')}"
                          
                artifacts.append(Document(
                    page_content=content,
                    metadata={
                        "source": "met_museum", 
                        "title": obj_data.get('title', 'N/A'), 
                        "url": obj_data.get('objectURL', obj_url),
                        "image_url": obj_data.get('primaryImageSmall')
                    }
                ))
            except Exception:
                continue # Skip if a single object fails
        return artifacts
    except Exception as e:
        st.error(f"Error fetching from Met Museum: {e}")
        return []

# --- 2. Function to Create RAG Chain ---

@st.cache_resource(show_spinner="Setting up RAG chain...")
def create_rag_chain(documents: list[Document]):
    """Creates a RAG chain with FAISS, local embeddings, and Llama 3 via Groq."""
    
    # 1. Create Embeddings
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # 2. Create Vector Store (FAISS)
    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
    except ValueError as e:
        st.error(f"Error creating vector store: {e}. This might happen if no documents were found.")
        return None
            
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # 3. Define LLM (Llama 3 via Groq)
    # This securely reads the API key from your .streamlit/secrets.toml file
    try:
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant", 
            groq_api_key=st.secrets["GROQ_API_KEY"]
        )
    except Exception as e:
        st.error(f"Error initializing Groq LLM: {e}. Have you set your GROQ_API_KEY in .streamlit/secrets.toml?")
        return None
    
    # 4. Define Prompt Template
    template = """
    You are a "Cultural Heritage Explorer" assistant. 
    Your task is to synthesize information from the provided context (from Wikipedia and museum archives) to answer the user's query.
    
    Rules:
    1. Base your answer *only* on the provided context.
    2. If the context is insufficient, state that you cannot answer the question with the provided information.
    3. Be detailed, insightful, and structured. Use markdown for formatting.
    4. When you use information, cite the source title.

    CONTEXT:
    {context}

    
    USER'S QUERY:
    {question}

    YOUR ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # 5. Helper function to format context
    def format_docs(docs):
        return "\n\n".join(
            f"--- Source: {doc.metadata['title']} ---\n{doc.page_content}" 
            for doc in docs
        )

    # 6. Build the RAG Chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

# --- 3. Streamlit UI ---

st.set_page_config(page_title="Cultural Heritage Explorer", layout="wide")
st.title("🏛️ Cultural Heritage Explorer (RAG)")
st.markdown("Powered by `Llama 3 (Groq)`, `LangChain`, `FAISS`, `Wikipedia`, & `The Met Museum`")

# User Input
query = st.text_input("Explore global heritage (e.g., 'Rosetta Stone', 'Vermeer', 'Japanese armor'):", key="query_input")

if st.button("Explore", type="primary"):
    if not query:
        st.warning("Please enter a query.")
    else:
        # 1. Retrieve Data
        wiki_docs = get_wikipedia_data(query)
        met_docs = get_met_museum_data(query)
        all_docs = wiki_docs + met_docs
        
        if not all_docs:
            st.error("No information found for that query. Please try another term.")
        else:
            st.success(f"Found {len(wiki_docs)} Wikipedia articles and {len(met_docs)} museum artifacts.")
            
            # 2. Create RAG Chain
            rag_chain = create_rag_chain(all_docs)
            
            if rag_chain:
                # 3. Generate Response
                st.subheader("Generated Insights (from Llama 3 on Groq):")
                with st.chat_message("ai"):
                    response_stream = rag_chain.stream(query)
                    st.write_stream(response_stream)
                
                # 4. Display Citations / Sources
                st.subheader("Retrieved Context (Sources):")
                
                # Dynamically create columns
                num_columns = min(len(all_docs), 3) # Max 3 columns
                if num_columns > 0:
                    cols = st.columns(num_columns)
                    col_index = 0
                    for doc in all_docs:
                        with cols[col_index % num_columns]:
                            with st.container(border=True):
                                st.markdown(f"**{doc.metadata['title']}**")
                                st.caption(f"Source: {doc.metadata['source']}")
                                st.link_button("View Source", doc.metadata['url'])
                                
                                if doc.metadata.get('image_url'):
                                    st.image(doc.metadata['image_url'], width=150)
                        col_index += 1