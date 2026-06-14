import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# 1. Page Configuration
st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🤖", layout="centered")
st.title("🤖 Zyro Dynamics HR Help Desk")
st.write("Welcome! Ask any questions regarding Zyro Dynamics HR policies.")

# 2. Initialize RAG Components (Cached to run only once)
@st.cache_resource
def initialize_rag():
    # Load Documents
    corpus_path = "./zyro-dynamics-hr-corpus"
    loader = PyPDFDirectoryLoader(corpus_path)
    documents = loader.load()
    
    # Chunk Documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    # Embeddings & Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    # Initialize LLM (Groq)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=512, api_key=st.secrets["GROQ_API_KEY"])
    
    # Prompts
    RAG_PROMPT = ChatPromptTemplate.from_template("""
    You are an HR Help Desk assistant for Zyro Dynamics. Answer the question using ONLY the provided context. 
    If the context contains information that is equivalent or closely related to the question, use it to answer directly. 
    Do not say information is missing if the answer can be reasonably inferred from the context. 
    If the answer is truly not present in the context, reply exactly: "I could not find this information in the Zyro Dynamics HR policy documents."
    
    Context: {context}
    Question: {question}
    Answer: 
    """)
    
    OOS_PROMPT = ChatPromptTemplate.from_template("""
    You are an AI gatekeeper for the Zyro Dynamics HR Help Desk. Your job is to determine if a user's question can be answered by our specific set of HR policy documents. 
    Our official HR corpus only covers the following topics: 
    1. Company Profile, 2. Employee Handbook, 3. Leave Policy, 4. Work From Home Policy, 5. Code of Conduct, 
    7. Performance Review Policy, 8. Compensation & Benefits Policy, 9. IT & Data Security Policy, 
    10. Prevention of Sexual Harassment Policy, 11. Onboarding and Separation Policy, 12. Travel & Expense Policy.
    
    Respond with exactly one word:
    - 'YES' if the question is directly related to any of the 12 HR policy topics listed above.
    - 'NO' if the question is a general greeting, unrelated technical support, programming, math, or completely outside Zyro Dynamics HR.
    
    Question: {question}
    Classification:
    """)
    
    return retriever, llm, RAG_PROMPT, OOS_PROMPT

# Run initialization
try:
    retriever, llm, RAG_PROMPT, OOS_PROMPT = initialize_rag()
except Exception as e:
    st.error(f"Failed to initialize RAG system: {e}")
    st.stop()

# 3. Chat History Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Chat Input & Processing
if user_query := st.chat_input("Type your HR question here..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # Generate bot response
    with st.chat_message("assistant"):
        with st.spinner("Searching internal policies..."):
            try:
                # Step A: Guardrail Check
                guardrail_chain = OOS_PROMPT | llm | StrOutputParser()
                classification = guardrail_chain.invoke({"question": user_query}).strip().upper()
                
                if "NO" in classification:
                    response_text = "I could not find this information in the Zyro Dynamics HR policy documents."
                else:
                    # Step B: Standard RAG Retrieval & Inference
                    docs = retriever.invoke(user_query)
                    context = "\n\n".join(doc.page_content for doc in docs)
                    
                    rag_chain = RAG_PROMPT | llm | StrOutputParser()
                    response_text = rag_chain.invoke({"context": context, "question": user_query})
                
                # Render response
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)