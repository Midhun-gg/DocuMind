import chromadb
from chromadb.db.migrations import InconsistentHashError
import shutil
import time
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import os

class VectorStoreManager:
    """
    Manages vector database operations using ChromaDB and Sentence Transformers
    """
    
    def __init__(self, collection_name: str = "documind_collection"):
        """
        Initialize vector store manager
        
        Args:
            collection_name: Name of ChromaDB collection
        """
        self.collection_name = collection_name
        
        # Initialize ChromaDB client (persistent storage) with auto-recovery
        db_path = "./chroma_db"
        try:
            self.client = chromadb.PersistentClient(path=db_path)
        except InconsistentHashError:
            # Database migrations inconsistent: reset local Chroma store
            try:
                shutil.rmtree(db_path, ignore_errors=True)
            except Exception:
                pass
            try:
                self.client = chromadb.PersistentClient(path=db_path)
            except InconsistentHashError:
                # As a last resort, create a fresh directory with a unique suffix
                fresh_path = f"./chroma_db_{int(time.time())}"
                os.makedirs(fresh_path, exist_ok=True)
                self.client = chromadb.PersistentClient(path=fresh_path)
        except Exception:
            # Fallback: attempt clean re-init if other startup errors occur
            try:
                shutil.rmtree(db_path, ignore_errors=True)
            except Exception:
                pass
            try:
                self.client = chromadb.PersistentClient(path=db_path)
            except Exception:
                fresh_path = f"./chroma_db_{int(time.time())}"
                os.makedirs(fresh_path, exist_ok=True)
                self.client = chromadb.PersistentClient(path=fresh_path)
        
        # Initialize embedding model (local, no API needed)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text using Sentence Transformers
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    def create_vector_store(self, chunks: List[Dict[str, Any]]) -> Any:
        """
        Create vector store from document chunks
        
        Args:
            chunks: List of document chunks with metadata
            
        Returns:
            ChromaDB collection
        """
        # Extract texts and metadata
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [chunk['metadata']['chunk_id'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        return self.collection
    
    def search_similar(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Search for similar documents using semantic search
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])[0]
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
        
        return formatted_results
    
    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        Add new documents to existing collection
        
        Args:
            chunks: List of document chunks with metadata
        """
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [chunk['metadata']['chunk_id'] for chunk in chunks]
        
        embeddings = self.generate_embeddings(texts)
        
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def clear_vector_store(self):
        """
        Clear all data from vector store
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"Error clearing vector store: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            'total_documents': count,
            'collection_name': self.collection_name
        }

