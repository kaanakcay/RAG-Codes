import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

#Define persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metada")

#define embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#Load the existing vector database with embedding func
db = Chroma(embedding_function=embeddings, persist_directory=persistent_directory)

query = "How did Juliet die?"

#Retrieve relevant documents based on the query
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":3, "score_threshold":0.1})

relevant_docs = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")
