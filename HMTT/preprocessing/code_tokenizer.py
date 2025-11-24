"""
Code Tokenizer for HMTT.

Dynamic code tokenizer that learns patterns from corpus without hardcoding.
Uses AST-based analysis and statistical pattern recognition.
"""

import re
from typing import List, Optional
from .dynamic_analyzer import DynamicCodeAnalyzer


class CodeTokenizer:
    """
    Dynamic code tokenizer that learns from corpus.
    
    Extracts atomic units:
    - Identifiers
    - Keywords (learned)
    - Operators
    - Literals (strings, numbers)
    - Punctuation
    
    Falls back to regex-based tokenization if tree-sitter is not available.
    """
    
    def __init__(self, language: str = "python", corpus_samples: Optional[List[str]] = None):
        """
        Initialize the code tokenizer with dynamic learning.
        
        Args:
            language: Programming language hint ("python", "javascript", "c", etc.)
            corpus_samples: Optional corpus to learn patterns from
        """
        self.language = language
        self.tree_sitter_available = False
        
        # Use dynamic analyzer
        self.analyzer = DynamicCodeAnalyzer()
        
        if corpus_samples:
            # Learn from provided corpus
            patterns = self.analyzer.analyze_corpus(corpus_samples, language)
            self.KEYWORDS = patterns['keywords']
            self.BUILTINS = patterns['builtins']
            self.CONTROL_FLOW = patterns['control_flow']
        else:
            # Start with empty sets - will learn on-the-fly
            self.KEYWORDS = set()
            self.BUILTINS = set()
            self.CONTROL_FLOW = set()
        
        # Try to import tree-sitter
        try:
            import tree_sitter
            self.tree_sitter_available = True
            self._init_tree_sitter(language)
        except ImportError:
            pass
    
    def _init_tree_sitter(self, language: str):
        """Initialize tree-sitter parser for the given language."""
        try:
            import tree_sitter
            
            # This is a placeholder - in production, you'd need to:
            # 1. Clone tree-sitter language repositories
            # 2. Build the languages
            # 3. Load the compiled libraries
            
            # For now, we'll use the fallback tokenizer
            self.tree_sitter_available = False
            
        except Exception:
            self.tree_sitter_available = False
    
    def tokenize(self, code_text: str, learn: bool = True) -> List[str]:
        """
        Tokenize code into atomic units.
        
        Args:
            code_text: Source code text
            learn: Whether to learn new patterns from this text
            
        Returns:
            List of atomic token strings
        """
        if self.tree_sitter_available:
            return self._tokenize_with_tree_sitter(code_text, learn)
        else:
            return self._tokenize_with_regex(code_text, learn)
    
    def _tokenize_with_tree_sitter(self, code_text: str, learn: bool = True) -> List[str]:
        """
        Tokenize using tree-sitter AST.
        
        Args:
            code_text: Source code text
            learn: Whether to learn patterns
            
        Returns:
            List of tokens
        """
        # This would use tree-sitter to parse and extract tokens
        # For now, fall back to regex
        return self._tokenize_with_regex(code_text, learn)
    
    def _tokenize_with_regex(self, code_text: str, learn: bool = True) -> List[str]:
        """
        Tokenize using regex patterns (fallback method).
        
        Args:
            code_text: Source code text
            learn: Whether to learn new patterns
            
        Returns:
            List of tokens
        """
        tokens = []
        i = 0
        
        while i < len(code_text):
            # Skip whitespace
            if code_text[i].isspace():
                i += 1
                continue
            
            # String literals (double quotes)
            if code_text[i] == '"':
                string_end = self._find_string_end(code_text, i, '"')
                tokens.append(code_text[i:string_end])
                i = string_end
                continue
            
            # String literals (single quotes)
            if code_text[i] == "'":
                string_end = self._find_string_end(code_text, i, "'")
                tokens.append(code_text[i:string_end])
                i = string_end
                continue
            
            # Comments (single line)
            if code_text[i:i+2] in ['//', '#']:
                comment_end = code_text.find('\n', i)
                if comment_end == -1:
                    comment_end = len(code_text)
                tokens.append(code_text[i:comment_end])
                i = comment_end
                continue
            
            # Comments (multi-line)
            if code_text[i:i+2] == '/*':
                comment_end = code_text.find('*/', i + 2)
                if comment_end == -1:
                    comment_end = len(code_text)
                else:
                    comment_end += 2
                tokens.append(code_text[i:comment_end])
                i = comment_end
                continue
            
            # Numbers (including hex, binary, floats)
            if code_text[i].isdigit():
                num_match = re.match(
                    r'0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+|\d+\.?\d*(?:[eE][+-]?\d+)?',
                    code_text[i:]
                )
                if num_match:
                    tokens.append(num_match.group(0))
                    i += num_match.end()
                    continue
            
            # Identifiers and keywords
            if code_text[i].isalpha() or code_text[i] == '_':
                ident_match = re.match(r'[a-zA-Z_]\w*', code_text[i:])
                if ident_match:
                    token = ident_match.group(0)
                    tokens.append(token)
                    # Learn if it looks like a keyword
                    if learn and self.analyzer._looks_like_keyword(token):
                        self.KEYWORDS.add(token)
                    i += ident_match.end()
                    continue
            
            # Multi-character operators
            if i + 2 < len(code_text):
                three_char = code_text[i:i+3]
                if three_char in ['===', '!==', '>>>', '<<=', '>>=', '**=', '//=']:
                    tokens.append(three_char)
                    i += 3
                    continue
            
            if i + 1 < len(code_text):
                two_char = code_text[i:i+2]
                if two_char in ['==', '!=', '<=', '>=', '<<', '>>', '&&', '||',
                               '++', '--', '+=', '-=', '*=', '/=', '%=', '&=',
                               '|=', '^=', '->', '=>', '::']:
                    tokens.append(two_char)
                    i += 2
                    continue
            
            # Single character operators and punctuation
            if code_text[i] in '+-*/%=<>!&|^~()[]{}.,;:?@#$`\\':
                tokens.append(code_text[i])
                i += 1
                continue
            
            # Any other character
            tokens.append(code_text[i])
            i += 1
        
        return tokens
    
    def _find_string_end(self, text: str, start: int, quote: str) -> int:
        """
        Find the end of a string literal, handling escape sequences.
        
        Args:
            text: Full text
            start: Index of opening quote
            quote: Quote character (' or ")
            
        Returns:
            Index after closing quote
        """
        i = start + 1
        
        while i < len(text):
            if text[i] == '\\' and i + 1 < len(text):
                # Skip escaped character
                i += 2
            elif text[i] == quote:
                return i + 1
            else:
                i += 1
        
        # Unclosed string
        return len(text)
    
    def is_keyword(self, token: str) -> bool:
        """
        Check if a token is a keyword (using learned patterns).
        
        Args:
            token: Token string
            
        Returns:
            True if token is a keyword
        """
        return token in self.KEYWORDS or self.analyzer.is_keyword(token)
    
    def is_identifier(self, token: str) -> bool:
        """
        Check if a token is an identifier.
        
        Args:
            token: Token string
            
        Returns:
            True if token is an identifier
        """
        return re.match(r'^[a-zA-Z_]\w*$', token) is not None and token not in self.KEYWORDS
    
    def is_literal(self, token: str) -> bool:
        """
        Check if a token is a literal (using dynamic analysis).
        
        Args:
            token: Token string
            
        Returns:
            True if token is a literal
        """
        return self.analyzer.is_literal(token)
