"""
HMTT Encoder.

Encodes text into token IDs using the trained HMTT tokenizer.
"""

from typing import List, Optional, Dict
from pathlib import Path

try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False

from ..preprocessing.partitioner import TextPartitioner
from ..preprocessing.math_tokenizer import MathTokenizer
from ..preprocessing.code_tokenizer import CodeTokenizer
from ..preprocessing.nl_tokenizer import NLTokenizer


class HMTTEncoder:
    """
    HMTT Encoder: Text → Token IDs
    
    Process:
    1. Partition text into NL/MATH/CODE regions
    2. Apply domain-specific pre-tokenization
    3. Apply BPE merges
    4. Convert to token IDs
    
    Guarantees:
    - Atomicity of math commands, variables, and numbers
    - Atomicity of code identifiers, keywords, and literals
    - No splitting of formal symbols
    """
    
    def __init__(
        self,
        tokenizer_path: str,
        code_language: str = "python",
        use_unicode_nl: bool = True
    ):
        """
        Initialize the HMTT encoder.
        
        Args:
            tokenizer_path: Path to trained tokenizer file
            code_language: Programming language for code tokenizer
            use_unicode_nl: Use unicode-aware NL tokenization
        """
        if not TOKENIZERS_AVAILABLE:
            raise ImportError(
                "HuggingFace tokenizers not available. "
                "Install with: pip install tokenizers"
            )
        
        # Load trained tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Initialize pre-tokenizers
        self.partitioner = TextPartitioner()
        self.math_tokenizer = MathTokenizer()
        self.code_tokenizer = CodeTokenizer(language=code_language)
        self.nl_tokenizer = NLTokenizer(use_unicode=use_unicode_nl)
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Add <s> and </s> tokens
            
        Returns:
            List of token IDs
        """
        # Get pre-tokens
        pre_tokens = self._pre_tokenize(text)
        
        # Join pre-tokens with spaces
        pre_tokenized_text = ' '.join(pre_tokens)
        
        # Apply BPE and convert to IDs
        encoding = self.tokenizer.encode(pre_tokenized_text)
        ids = encoding.ids
        
        # Add special tokens if requested
        if add_special_tokens:
            bos_id = self.tokenizer.token_to_id("<s>")
            eos_id = self.tokenizer.token_to_id("</s>")
            
            if bos_id is not None and eos_id is not None:
                ids = [bos_id] + ids + [eos_id]
        
        return ids
    
    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True
    ) -> List[List[int]]:
        """
        Encode multiple texts to token IDs.
        
        Args:
            texts: List of input texts
            add_special_tokens: Add <s> and </s> tokens
            
        Returns:
            List of token ID lists
        """
        return [self.encode(text, add_special_tokens) for text in texts]
    
    def _pre_tokenize(self, text: str) -> List[str]:
        """
        Pre-tokenize text using domain-specific tokenizers.
        
        Args:
            text: Input text
            
        Returns:
            List of pre-tokens
        """
        # Partition text into regions
        regions = self.partitioner.partition_to_dict(text)
        
        # Tokenize each region
        all_tokens = []
        
        for region in regions:
            region_type = region["type"]
            region_text = region["text"]
            
            if region_type == "math":
                tokens = self.math_tokenizer.tokenize(region_text)
            elif region_type == "code":
                tokens = self.code_tokenizer.tokenize(region_text)
            else:  # nl
                tokens = self.nl_tokenizer.tokenize(region_text)
            
            all_tokens.extend(tokens)
        
        return all_tokens
    
    def get_vocab(self) -> Dict[str, int]:
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
    
    def token_to_id(self, token: str) -> Optional[int]:
        """
        Convert token to ID.
        
        Args:
            token: Token string
            
        Returns:
            Token ID or None if not in vocabulary
        """
        return self.tokenizer.token_to_id(token)
    
    def id_to_token(self, id: int) -> Optional[str]:
        """
        Convert ID to token.
        
        Args:
            id: Token ID
            
        Returns:
            Token string or None if invalid ID
        """
        return self.tokenizer.id_to_token(id)
    
    def encode_with_offsets(self, text: str) -> tuple[List[int], List[tuple[int, int]]]:
        """
        Encode text and return token offsets.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (token IDs, list of (start, end) character offsets)
        """
        pre_tokens = self._pre_tokenize(text)
        pre_tokenized_text = ' '.join(pre_tokens)
        
        encoding = self.tokenizer.encode(pre_tokenized_text)
        
        return encoding.ids, encoding.offsets
    
    def count_tokens(self, text: str) -> int:
        """
        Count number of tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return len(self.encode(text, add_special_tokens=False))
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Get token strings (without IDs).
        
        Args:
            text: Input text
            
        Returns:
            List of token strings
        """
        pre_tokens = self._pre_tokenize(text)
        pre_tokenized_text = ' '.join(pre_tokens)
        
        encoding = self.tokenizer.encode(pre_tokenized_text)
        
        return encoding.tokens
