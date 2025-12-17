"""PDF Processing Service"""
import os
from PyPDF2 import PdfReader
from typing import Dict, List
from src.config import Config
import logging
import uuid

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.upload_folder = Config.PDF_UPLOAD_FOLDER
        os.makedirs(self.upload_folder, exist_ok=True)
    
    def process_pdf(self, file_path: str) -> Dict:
        """Extract text from PDF"""
        try:
            reader = PdfReader(file_path)
            text_parts = []
            page_count = len(reader.pages)
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    text_parts.append({
                        'page': page_num,
                        'content': text.strip()
                    })
            
            full_text = '\n\n'.join([part['content'] for part in text_parts])
            
            return {
                'text': full_text,
                'page_count': page_count,
                'pages': text_parts,
                'success': True
            }
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return {
                'text': '',
                'page_count': 0,
                'pages': [],
                'success': False,
                'error': str(e)
            }
    
    def save_pdf(self, file_content: bytes, filename: str) -> str:
        """Save uploaded PDF file"""
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(filename)[1]
        file_path = os.path.join(self.upload_folder, f"{file_id}{file_ext}")
        
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        return file_path
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into chunks for embedding"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks

