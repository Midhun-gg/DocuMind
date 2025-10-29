import os
from typing import List, Dict
from PyPDF2 import PdfReader
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """
    Handles document loading, text extraction, and chunking
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks for context preservation
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\\n\\n", "\\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, file_path: str) -> List[Dict[str, any]]:
        """
        Extract text from PDF file with page information
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of dictionaries containing text and metadata for each page
        """
        pages_data = []
        try:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    pages_data.append({
                        'text': text,
                        'page': page_num,
                        'total_pages': len(reader.pages)
                    })
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
        
        return pages_data
    
    def extract_text_from_docx(self, file_path: str) -> List[Dict[str, any]]:
        """
        Extract text from DOCX file
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            List containing document text and metadata
        """
        try:
            doc = docx.Document(file_path)
            text = "\\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
            
            return [{
                'text': text,
                'page': 1,
                'total_pages': 1
            }]
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
    
    def extract_text_from_txt(self, file_path: str) -> List[Dict[str, any]]:
        """
        Extract text from TXT file
        
        Args:
            file_path: Path to TXT file
            
        Returns:
            List containing text and metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            return [{
                'text': text,
                'page': 1,
                'total_pages': 1
            }]
        except Exception as e:
            raise Exception(f"Error reading TXT: {str(e)}")
    
    def process_document(self, file_path: str, filename: str) -> List[Dict[str, any]]:
        """
        Process document: extract text, create chunks with metadata
        
        Args:
            file_path: Path to document file
            filename: Original filename
            
        Returns:
            List of chunks with metadata
        """
        # Determine file type and extract text
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext == '.pdf':
            pages_data = self.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            pages_data = self.extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            pages_data = self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Create chunks with metadata
        all_chunks = []
        for page_data in pages_data:
            text = page_data['text']
            page_num = page_data['page']
            
            # Split text into chunks
            text_chunks = self.text_splitter.split_text(text)
            
            # Add metadata to each chunk
            for chunk_idx, chunk in enumerate(text_chunks):
                all_chunks.append({
                    'text': chunk,
                    'metadata': {
                        'document': filename,
                        'page': page_num,
                        'chunk_id': f"{filename}_page{page_num}_chunk{chunk_idx}",
                        'source_type': file_ext[1:]  # Remove the dot
                    }
                })
        
        return all_chunks
    
    def get_document_stats(self, chunks: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Get statistics about processed document
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Dictionary with document statistics
        """
        total_chars = sum(len(chunk['text']) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        
        return {
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'avg_chunk_size': avg_chunk_size,
            'documents': len(set(chunk['metadata']['document'] for chunk in chunks))
        }

