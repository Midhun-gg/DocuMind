import os
import json
import subprocess
from typing import List, Dict, Tuple, Any, Optional

class LLMHandler:
    """
    Handles LLM interactions for generating answers using llama3.1:8b
    """
    
    def __init__(self, model: str = "llama3.1:8b", ollama_python_path: Optional[str] = None):
        """
        Initialize LLM handler with llama3.1:8b via Ollama
        
        Args:
            model: Model name (default: llama3.1:8b)
        """
        self.model = model
        self.ollama_error_message = ""
        self.ollama_python = ollama_python_path or os.environ.get("OLLAMA_PY", "")
        self.ollama_available = False

        # Validate the secondary venv Python by calling worker with --check
        if self.ollama_python:
            try:
                result = subprocess.run(
                    [self.ollama_python, os.path.join(os.path.dirname(__file__), "ollama_worker.py"), "--check"],
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
                ok = False
                if result.stdout:
                    try:
                        data = json.loads(result.stdout.strip())
                        ok = bool(data.get("ok"))
                        if not ok:
                            self.ollama_error_message = data.get("error", "Unknown error")
                    except Exception:
                        self.ollama_error_message = result.stdout.strip()
                if result.stderr and not ok:
                    self.ollama_error_message = (self.ollama_error_message + "\n" + result.stderr.strip()).strip()
                self.ollama_available = ok
                if not ok and not self.ollama_error_message:
                    self.ollama_error_message = "Ollama worker check failed."
            except Exception as e:
                self.ollama_available = False
                self.ollama_error_message = f"Failed to execute Ollama worker: {str(e)}"
        else:
            self.ollama_error_message = (
                "Secondary venv Python not configured. Set OLLAMA_PY to the python.exe "
                "of the Ollama venv (e.g., C:\\GIT\\DocuMind\\.venv_ollama\\Scripts\\python.exe)."
            )
    
    def create_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """
        Create context from relevant documents
        
        Args:
            relevant_docs: List of relevant document chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            metadata = doc['metadata']
            text = doc['text']
            
            context_part = f"""
Source {i} (Document: {metadata['document']}, Page: {metadata['page']}):
{text}
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def generate_answer(
        self, 
        query: str, 
        relevant_docs: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate answer using Llama 3.2 based on retrieved documents
        
        Args:
            query: User query
            relevant_docs: List of relevant document chunks
            
        Returns:
            Tuple of (answer, sources)
        """
        # Early return if Ollama isn't available
        if not self.ollama_available:
            answer = (
                "LLM is unavailable. "
                + self.ollama_error_message
            )
            sources: List[Dict[str, Any]] = []
            for doc in relevant_docs:
                sources.append({
                    'document': doc['metadata']['document'],
                    'page': doc['metadata'].get('page', 'N/A'),
                    'text': doc['text']
                })
            return answer, sources

        # Create context from relevant documents
        context = self.create_context(relevant_docs)
        
        # Create system prompt
        system_prompt = """You are an intelligent document assistant. Your task is to answer questions based ONLY on the provided context from documents.

Guidelines:
1. Answer the question using only the information from the provided sources
2. Be concise and accurate
3. If the context doesn't contain relevant information, say so
4. Reference specific sources when making claims (e.g., "According to Source 1...")
5. Maintain a professional and helpful tone
6. Do not make up information or use knowledge outside the provided context
"""
        
        # Create user prompt
        user_prompt = f"""Context from documents:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above. Reference the sources you use."""
        
        # Generate response via subprocess worker in secondary venv
        try:
            result = subprocess.run(
                [
                    self.ollama_python,
                    os.path.join(os.path.dirname(__file__), "ollama_worker.py"),
                    "--model", self.model,
                    "--mode", "chat",
                    "--system", system_prompt,
                    "--user", user_prompt,
                    "--temperature", str(0.7),
                    "--num_predict", str(500),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            answer = ""
            if result.stdout:
                try:
                    payload = json.loads(result.stdout.strip())
                    if payload.get("ok"):
                        answer = payload.get("answer", "")
                    else:
                        answer = f"Error generating response: {payload.get('error', 'Unknown error')}"
                except Exception:
                    answer = result.stdout.strip()
            if not answer and result.stderr:
                answer = f"Error generating response: {result.stderr.strip()}"
            if not answer:
                answer = "No response from Ollama worker."
        except Exception as e:
            answer = f"Error generating response: {str(e)}\nPlease ensure the Ollama worker venv is configured."
        
        # Format sources
        sources: List[Dict[str, Any]] = []
        for doc in relevant_docs:
            sources.append({
                'document': doc['metadata']['document'],
                'page': doc['metadata'].get('page', 'N/A'),
                'text': doc['text']
            })
        
        return answer, sources
    
    def generate_summary(self, text: str) -> str:
        """
        Generate a summary of the provided text using Llama 3.2
        
        Args:
            text: Text to summarize
            
        Returns:
            Summary text
        """
        if not self.ollama_available:
            return (
                "LLM is unavailable. "
                + self.ollama_error_message
            )

        try:
            result = subprocess.run(
                [
                    self.ollama_python,
                    os.path.join(os.path.dirname(__file__), "ollama_worker.py"),
                    "--model", self.model,
                    "--mode", "summary",
                    "--system", "You are a helpful assistant that creates concise summaries.",
                    "--user", f"Please provide a concise summary of the following text:\n\n{text}",
                    "--temperature", str(0.5),
                    "--num_predict", str(200),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.stdout:
                try:
                    payload = json.loads(result.stdout.strip())
                    if payload.get("ok"):
                        return payload.get("answer", "")
                    return f"Error generating summary: {payload.get('error', 'Unknown error')}"
                except Exception:
                    return result.stdout.strip()
            if result.stderr:
                return f"Error generating summary: {result.stderr.strip()}"
            return "No response from Ollama worker."
        except Exception as e:
            return f"Error generating summary: {str(e)}\nPlease ensure the Ollama worker venv is configured."
