from langchain_ollama import ChatOllama,OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.vectorstores import InMemoryVectorStore
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

file_path="C:/Users/SK/Downloads/AI_ML Engineer_JD.pdf"
data=PyMuPDFLoader(file_path)


doc=data.load()
doc[0]

text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=10)
texts=text_splitter.split_documents(doc)
# print(texts)

embeddings=OllamaEmbeddings(
    model="llama3"
)

embedd_dim=len(embeddings.embed_query("test"))
index=faiss.IndexFlatL2(embedd_dim)

vectorestore=FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

vectorestore.add_documents(texts)

# vectorstore=InMemoryVectorStore.from_documents(
#     texts,
#     embedding=embeddings
# )

query=input("Ask your questions: ")

retriever=vectorestore.as_retriever()

retrieved_documents=retriever.invoke(query)

print(retrieved_documents[0].page_content)


#without RAG SYSTEM

llm=ChatOllama(
    model="Qwen2.5:latest",
    temperature=0.7,
    top_k=1
)

message=[
    ("system",
     "your helful assistent response the user QA"),
     ("human",
      query)
]

ok=llm.invoke(message)

print("Without RAG response: ",ok.content)