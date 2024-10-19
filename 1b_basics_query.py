
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

#Önce database in pathini alıyorum
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

#Embeddinge ihtiyacım var çünkü soruyu da embed etmem lazım ki similarity check yapılsın
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,embedding_function=embeddings)

#User question
query = "Who is Odysseus' wife?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'k':3, "score_threshold":0.4 }# k=knn -- en yakın 3 resultu getirecek retriever
)

#taking relevent docs from retirever
relevant_docs = retriever.invoke(query) #query ile relevant olan documentleri getiriyor

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1): #YUKARIDA K YI KAÇ VERİRSEK O KADAR DOCUMENT PRİNT EDER. BU ÖRNEKTE 3
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")





