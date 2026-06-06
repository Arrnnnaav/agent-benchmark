import os
import uuid # every doc stored in ChromaDB needs a unique ID. so uuid generated them
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions 

load_dotenv()

class ChromaMemoryStore: 
    def __init__(self):
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
        #create a chromadb client that saves data to disk at persist_dir
        self.client = chromadb.PersistentClient(path= persist_dir) 
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name = "all-MiniLM-L6-v2"
        )
        #get the collection name "agent_memory" if it already exists or create it fresh if it doesnt 
        self.collection = self.client.get_or_create_collection(
            name = "agent_memory",
            embedding_function = self.embedding_fn
        )
    
    def save(self, text: str, metadata: dict = None) -> str:
        doc_id = str(uuid.uuid4()) #generate a unique ID for this document
        #Store the document. 
        # Three parallel lists — documents, ids, metadatas — must all be the same length. 
        # We pass single-item lists because add supports batch operations (adding many at once), so it always expects lists even for one item. ChromaDB automatically calls our embedding_fn on each document before storing — we never touch vectors directly. 
        self.collection.add(documents = [text],
                             ids = [doc_id],
                             metadatas = [metadata or {}])
        return doc_id
    
    def search(self, query: str, n_results: int = 3) -> list[str]:
        if self.collection.count() == 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count())
        )
        documents = results.get("documents", [[]])
        if not documents or not documents[0]:
            return []
        return documents[0]
    
    def count(self) -> int: # how many documents are stored
        return self.collection.count()
    
    def clear(self) -> None: #delete everything in memory
        #delete the entire collection
        self.client.delete_collection("agent_memory")
        #immediately recreate it empty
        self.collection = self.client.get_or_create_collection(
            name = "agent_memory",
            embedding_function= self.embedding_fn 
        )
