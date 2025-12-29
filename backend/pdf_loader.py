"""
PDF Loader module for extracting text from PDF files
"""

import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image


class PDFLoader:
    """
    Loads and extracts text from PDF files
    """
    
    def __init__(self):
        """Initialize the PDF loader"""
        pass
    
    def load_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF, with OCR fallback for image-based PDFs
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        text = ""
        
        # Try normal text extraction first
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error extracting text with pdfplumber: {e}")
        
        # If very little text extracted, use OCR
        if len(text.strip()) < 100:  # Threshold for "empty" PDF
            print(f"Using OCR for image-based PDF: {pdf_path}")
            try:
                images = convert_from_path(pdf_path)
                text = ""
                for i, image in enumerate(images):
                    print(f"Processing page {i + 1} with OCR...")
                    page_text = pytesseract.image_to_string(image)
                    if page_text:
                        text += page_text + "\n"
            except Exception as e:
                print(f"Error with OCR: {e}")
                raise
        
        if not text.strip():
            raise ValueError(f"No text could be extracted from PDF: {pdf_path}")
        
        return text
