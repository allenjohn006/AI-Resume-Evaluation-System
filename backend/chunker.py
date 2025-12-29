"""
Text Chunker module for splitting text into manageable pieces
"""


class TextChunker:
    """
    Handles chunking of text into smaller segments for processing
    """
    
    def __init__(self, chunk_size: int = 500):
        """
        Initialize the text chunker
        
        Args:
            chunk_size: Maximum number of words per chunk
        """
        self.chunk_size = chunk_size
    
    def chunk_text(self, text: str) -> list:
        """
        Split text into chunks by words
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size):
            chunks.append(" ".join(words[i:i + self.chunk_size]))

        return chunks
    
    def chunk_by_sentences(self, text: str) -> list:
        """
        Split text into chunks by sentences
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of sentence-based chunks
        """
        # TODO: Implement sentence-based chunking
        # Consider using nltk or spacy for better sentence detection
        
        return []
    
    def chunk_by_paragraphs(self, text: str) -> list:
        """
        Split text into chunks by paragraphs
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of paragraph-based chunks
        """
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
