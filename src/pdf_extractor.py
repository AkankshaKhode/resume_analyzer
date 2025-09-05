import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
from typing import Union
import io

class PDFExtractor:
    """Extract text from PDF files using PyMuPDF and pdfminer as fallback"""
    
    def __init__(self):
        pass
    
    def extract_text(self, pdf_file) -> str:
        """
        Extract text from uploaded PDF file
        
        Args:
            pdf_file: Streamlit uploaded file object
            
        Returns:
            str: Extracted text from PDF
        """
        try:
            # Try PyMuPDF first (faster)
            return self._extract_with_pymupdf(pdf_file)
        except Exception as e:
            print(f"PyMuPDF failed: {e}. Trying pdfminer...")
            try:
                # Fallback to pdfminer
                return self._extract_with_pdfminer(pdf_file)
            except Exception as e2:
                raise Exception(f"Both extraction methods failed. PyMuPDF: {e}, pdfminer: {e2}")
    
    def _extract_with_pymupdf(self, pdf_file) -> str:
        """
        Extract text using PyMuPDF
        """
        # Read the uploaded file
        pdf_bytes = pdf_file.read()
        
        # Open PDF from bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        text = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text()
        
        doc.close()
        return text.strip()
    
    def _extract_with_pdfminer(self, pdf_file) -> str:
        """
        Extract text using pdfminer (fallback)
        """
        # Reset file pointer
        pdf_file.seek(0)
        pdf_bytes = pdf_file.read()
        
        # Extract text using pdfminer
        text = extract_text(io.BytesIO(pdf_bytes))
        return text.strip()
    
    def validate_pdf(self, pdf_file) -> bool:
        """
        Validate if the uploaded file is a valid PDF
        """
        try:
            pdf_bytes = pdf_file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            doc.close()
            return True
        except:
            return False