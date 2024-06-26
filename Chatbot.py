__import__('pysqlite3')
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import streamlit as st
import File_QA, Chat_with_search, Chat_with_user_feedback
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI


if 'page' not in st.session_state:
    st.session_state.page = "home" 


def home_page():
    ## High functional materials retriever
    persist_directory_materials = "db_materials"
    embedding = OpenAIEmbeddings()
    vectordb_materials = Chroma(embedding_function=embedding,
                      persist_directory=persist_directory_materials)
  
    # retrieve information from the vector db
    retriever_materials = vectordb_materials.as_retriever() # search_kwargs={"k": 2} # search up to 2 most similar papers, default 4
    llm=OpenAI()
  
    # create the chain to answer questions
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(),
                                      chain_type="stuff",
                                      retriever=retriever_materials,
                                      return_source_documents=True)
    ## Cite sources
    def process_llm_response_old(llm_response):
      #print('---- Query ---')
      #print(llm_response['query'],'\n')
    
      #print('---- Answer ---')
      answer = llm_response['result']
      st.write(answer)
    
      #print('\n\n ---- Sources -----')
      for i, sources in enumerate(llm_response["source_documents"]):
          st.write(f"""
                       [{i}].\n
                       {sources.metadata['title']}\n
                       {sources.metadata['DOI']}\n\n
                    """)
    
    def process_llm_response(llm_response):
        st.markdown("## Answer")  # Header
        st.write(llm_response['result']) 
    
        st.markdown("## Sources")
    
        for i, source in enumerate(llm_response["source_documents"]):
            st.markdown(f"**Source {i+1}**") 
            st.write(f" * **Title:** {source.metadata['title']}")
            if 'DOI' in source.metadata:
                st.write(f" * **DOI:** {source.metadata['DOI']}") 
            st.markdown("---")  # Separator between sources
    
    
    # full Q&A
    st.title("💬 Chat with internal data")
    st.caption("🚀 A chatbot powered by vector database retrieval and OpenAI LLM")
    
    # Initialize session state for both conversational and QA history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    
    # Chat Interaction Loop
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    #tab1, tab2 = st.tabs(["General Functional Materials", "High Performance Metals"])
    # Check for different input types
    if query := st.text_input("Ask a question (or type something to chat):"):
        llm_response = qa_chain(query)
        process_llm_response(llm_response)


def files_page():
    st.title("Chat with Files")
    File_QA

def search_page():
    st.title("Chat with Search")
    Chat_with_search


def feedback_page():
    st.title("Chat with User Feedback")
    Chat_with_user_feedback


with st.sidebar:
    st.header("Apps")
    if st.button("Chat with Internal Data"):
        st.session_state.page = "home"
    if st.button("Chat with Search"):
        st.session_state.page = "search"    
    if st.button("Chat with User Feedback"):
        st.session_state.page = "feedback"
    if st.button("Chat with Files"):
        st.session_state.page = "files"


if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "files":
    files_page()
elif st.session_state.page == "search":
    search_page()
elif st.session_state.page == "feedback":
    feedback_page()
