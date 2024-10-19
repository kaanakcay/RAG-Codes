import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings




current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )
    

    #Reading content of the text
    loader = TextLoader(file_path)
    documents = loader.load() # now content of the text in documents variable

    #Lets split documents into chunks
    #text_splitter bölünmüş hali değil. Bu aslında bölücü fonksiyon.
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0) #CharacterTextSplitter fonksiyonu karakter karakter bölüyor documenti ve her 1000 karakterde bir chunk oluşacak. Bazı keliemeler göte gelebilir bence böyle çünkü bölünecekler sanki.
    docs = text_splitter.split_documents(documents)#bölme işlemini burada yapıyoruz

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")


    #Embedding de sıra
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")




