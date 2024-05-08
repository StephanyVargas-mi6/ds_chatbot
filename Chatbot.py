__import__('pysqlite3')
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import streamlit as st


## High functional materials retriever
persist_directory_materials = "db_materials"
embedding = OpenAIEmbeddings()
vectordb_materials = Chroma(embedding_function=embedding,
                  persist_directory=persist_directory_materials)
# retrieve information from the vector db
retriever_materials = vectordb_materials.as_retriever() # search_kwargs={"k": 2} # search up to 2 most similar papers, default 4

# full Q&A
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

llm=OpenAI()
# create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(),
                                  chain_type="stuff",
                                  retriever=retriever_materials,
                                  return_source_documents=True)
## Cite sources
def process_llm_response(llm_response):
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

# full Q&A

query = "what are high functional materials?"
llm_response = qa_chain(query)
process_llm_response(llm_response)
