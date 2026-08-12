import os
import chromadb
from sentence_transformers import SentenceTransformer

folder = os.path.dirname(os.path.abspath(__file__))

folder_path = os.path.join(folder, "docs")

files = os.listdir(folder_path)

#print(files)

chunks = []

for file in files:
    file_path1 = os.path.join(folder_path, file)
    name = os.path.basename(file_path1)
    with open(file_path1 , "r") as f:
        data = f.read()
        chunk = {
                 "document_id": name,
                 "Data": data
                }
        chunks.append(chunk)
#print(chunks)

Texts = []

for chunk in chunks:
    line = chunk["Data"]
    Texts.append(line)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(Texts)

print(embeddings.shape)

chrome_db_path = os.path.join(folder, "chroma_db")

client = chromadb.PersistentClient(path=chrome_db_path)

#collection = client.create_collection(name="Zepto_Policies")

collection = client.get_or_create_collection(
    name="Zepto_Policies",
    metadata={"hnsw:space": "cosine"}
)

print(collection)

for index, chunk in enumerate(chunks):

    # ID = chunk["document_id"]
    # Document = chunk["Data"]
    # Metadata = {"document_id" : chunk["document_id"]}
    # Embedding = embeddings[index]
    
    collection.add(

    ids = [chunk["document_id"]],
    documents = [chunk["Data"]],
    metadatas = [{"document_id" : chunk["document_id"]}],
    embeddings = [embeddings[index]]
        )
print(collection.count())