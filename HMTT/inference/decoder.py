"""
HMTT Decoder.

Decodes token IDs back to text using the trained HMTT tokenizer.
"""

from typing import List, Optional

try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False


class HMTTDecoder:
    """
    HMTT Decoder: Token IDs → Text
    
    Process:
    1. Convert token IDs to token strings
    2. Reconstruct text from tokens
    3. Preserve formatting and structure
    4. Recombine NL, Math, and Code regions
    """
    
    def __init__(self, tokenizer_path: str):
        """
        Initialize the HMTT decoder.
        
        Args:
            tokenizer_path: Path to trained tokenizer file
        """
        if not TOKENIZERS_AVAILABLE:
            raise ImportError(
                "HuggingFace tokenizers not available. "
                "Install with: pip install tokenizers"
            )
        
        # Load trained tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
    
    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Skip special tokens like <s>, </s>, <pad>
            
        Returns:
            Decoded text
        """
        # Use tokenizer's decode method
        text = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        
        return text
    
    def decode_batch(
        self,
        ids_list: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode multiple sequences of token IDs.
        
        Args:
            ids_list: List of token ID lists
            skip_special_tokens: Skip special tokens
            
        Returns:
            List of decoded texts
        """
        return [self.decode(ids, skip_special_tokens) for ids in ids_list]
    
    def decode_tokens(
        self,
        tokens: List[str]
    ) -> str:
        """
        Decode token strings to text.
        
        Args:
            tokens: List of token strings
            
        Returns:
            Decoded text
        """
        # Join tokens and clean up
        text = ''.join(tokens)
        
        # Basic cleanup
        text = self._post_process(text)
        
        return text
    
    def _post_process(self, text: str) -> str:
        """
        Post-process decoded text.
        
        Args:
            text: Decoded text
            
        Returns:
            Post-processed text
        """
        # Remove extra spaces before punctuation
        import re
        
        # Fix spaces before punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        # Fix spaces around parentheses and brackets
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)
        text = re.sub(r'\[\s+', '[', text)
        text = re.sub(r'\s+\]', ']', text)
        
        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def id_to_token(self, id: int) -> Optional[str]:
        """
        Convert token ID to token string.
        
        Args:
            id: Token ID
            
        Returns:
            Token string or None if invalid ID
        """
        return self.tokenizer.id_to_token(id)
    
    def token_to_id(self, token: str) -> Optional[int]:
        """
        Convert token string to ID.
        
        Args:
            token: Token string
            
        Returns:
            Token ID or None if not in vocabulary
        """
        return self.tokenizer.token_to_id(token)
    
    def get_vocab(self) -> dict:
        """
        Get vocabulary mapping.
        
        Returns:
            Dictionary of token -> ID
        """
        return self.tokenizer.get_vocab()
    
    def get_vocab_size(self) -> int:
        """
        Get vocabulary size.
        
        Returns:
            Size of vocabulary
        """
        return self.tokenizer.get_vocab_size()
    
    def decode_single_token(self, id: int) -> Optional[str]:
        """
        Decode a single token ID.
        
        Args:
            id: Token ID
            
        Returns:
            Decoded token string
        """
        return self.tokenizer.decode([id])
    
    def decode_with_ids(self, ids: List[int]) -> List[tuple[int, str]]:
        """
        Decode token IDs and return both IDs and tokens.
        
        Args:
            ids: List of token IDs
            
        Returns:
            List of (ID, token string) tuples
        """
        result = []
        
        for id in ids:
            token = self.id_to_token(id)
            if token is not None:
                result.append((id, token))
        
        return result
    
    def is_special_token(self, id: int) -> bool:
        """
        Check if token ID is a special token.
        
        Args:
            id: Token ID
            
        Returns:
            True if token is a special token
        """
        token = self.id_to_token(id)
        if token is None:
            return False
        
        special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
        return token in special_tokens
    
    def filter_special_tokens(self, ids: List[int]) -> List[int]:
        """
        Filter out special tokens from ID list.
        
        Args:
            ids: List of token IDs
            
        Returns:
            List of IDs with special tokens removed
        """
        return [id for id in ids if not self.is_special_token(id)]
