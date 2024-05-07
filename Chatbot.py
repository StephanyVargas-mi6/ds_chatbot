from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


## High functional materials retriever
persist_directory_materials = os.path.join(path,"db_materials")
embedding = OpenAIEmbeddings()
vectordb_materials = Chroma(embedding_function=embedding,
                  persist_directory=persist_directory_materials)
# retrieve information from the vector db
retriever_materials = vectordb_materials.as_retriever() # search_kwargs={"k": 2} # search up to 2 most similar papers, default 4
docs = retriever_materials.invoke("what are high functional materials?")

st.write(docs)
