"""
Natural Language Tokenizer for HMTT.

GPT-4 style regex-based tokenization for natural language text.
"""

import re
from typing import List


class NLTokenizer:
    """
    Natural language tokenizer using GPT-4 style regex patterns.
    
    Handles:
    - Contractions (don't, won't, etc.)
    - Punctuation splitting
    - Number preservation
    - Alphanumeric tokens
    """
    
    # Simpler ASCII-compatible pattern (default)
    SIMPLE_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)|"""  # Contractions
        r"""[a-zA-Z]+|"""  # Letters
        r"""\d+|"""  # Numbers
        r"""[^\s\w]+|"""  # Non-alphanumeric
        r"""\s+"""  # Whitespace
    )
    
    def __init__(self, use_unicode: bool = False):
        """
        Initialize the natural language tokenizer.
        
        Args:
            use_unicode: Use unicode-aware pattern (requires regex module)
        """
        self.use_unicode = use_unicode
        self.regex_available = False
        
        if use_unicode:
            try:
                import regex
                self.pattern = regex.compile(
                    r"""'(?:[sdmt]|ll|ve|re)|"""
                    r"""\p{L}+|"""
                    r"""\p{N}+|"""
                    r"""[^\s\p{L}\p{N}]+|"""
                    r"""\s+(?!\S)|"""
                    r"""\s+"""
                )
                self.regex_available = True
            except ImportError:
                # Fall back to simple pattern if regex module not available
                self.pattern = self.SIMPLE_PATTERN
        else:
            # Use simple ASCII-compatible pattern by default
            self.pattern = self.SIMPLE_PATTERN
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize natural language text.
        
        Args:
            text: Natural language text
            
        Returns:
            List of token strings
        """
        # Use regex pattern
        tokens = self.pattern.findall(text)
        
        # Filter out pure whitespace tokens (but keep them if they're significant)
        tokens = [t for t in tokens if t.strip() or t == ' ']
        
        return tokens
    
    def tokenize_with_spaces(self, text: str) -> List[str]:
        """
        Tokenize while preserving space tokens explicitly.
        
        Args:
            text: Natural language text
            
        Returns:
            List of token strings including space markers
        """
        tokens = self.pattern.findall(text)
        return tokens
    
    def preprocess_contractions(self, text: str) -> str:
        """
        Preprocess contractions to ensure consistent splitting.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized contractions
        """
        # Common contractions
        contractions = {
            "won't": "will not",
            "can't": "can not",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "'s": " is",  # or possessive - context dependent
        }
        
        result = text
        for contraction, expansion in contractions.items():
            result = result.replace(contraction, expansion)
        
        return result
    
    def split_punctuation(self, text: str) -> str:
        """
        Add spaces around punctuation for easier tokenization.
        
        Args:
            text: Input text
            
        Returns:
            Text with spaces around punctuation
        """
        # Add space before punctuation
        text = re.sub(r'([^\s\w])', r' \1 ', text)
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def is_number(self, token: str) -> bool:
        """
        Check if a token is a number.
        
        Args:
            token: Token string
            
        Returns:
            True if token is a number
        """
        try:
            float(token)
            return True
        except ValueError:
            return False
    
    def is_word(self, token: str) -> bool:
        """
        Check if a token is a word (alphabetic).
        
        Args:
            token: Token string
            
        Returns:
            True if token is a word
        """
        return token.isalpha()
    
    def is_punctuation(self, token: str) -> bool:
        """
        Check if a token is punctuation.
        
        Args:
            token: Token string
            
        Returns:
            True if token is punctuation
        """
        return bool(re.match(r'^[^\s\w]+$', token))
