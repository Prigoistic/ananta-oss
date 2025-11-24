"""
Dynamic content analyzer for HMTT tokenizers.

Uses NLP techniques and pattern analysis to automatically identify:
- Programming keywords without hardcoding
- Mathematical operators and symbols
- Code structure patterns
- LaTeX commands from corpus

No hardcoded lists - learns from content dynamically.
"""

import re
from typing import Set, List, Dict, Tuple, Optional
from collections import Counter, defaultdict


class DynamicMathAnalyzer:
    """
    Dynamically identifies LaTeX commands and math patterns from corpus.
    No hardcoded command lists - learns from actual usage.
    """
    
    def __init__(self):
        self.latex_command_pattern = re.compile(r'\\([a-zA-Z]+)\b')
        self.symbol_pattern = re.compile(r'[+\-*/=<>≤≥≠∈∉∪∩⊂⊃∀∃∇∂∫∑∏√±×÷]')
        self.learned_commands = Counter()
        self.learned_symbols = Counter()
        
    def analyze_corpus(self, texts: List[str], min_frequency: int = 2) -> Dict[str, Set[str]]:
        """
        Analyze a corpus of mathematical text to learn commands and patterns.
        
        Args:
            texts: List of math text samples
            min_frequency: Minimum occurrences to consider a pattern
            
        Returns:
            Dictionary with 'commands', 'symbols', 'operators'
        """
        for text in texts:
            # Extract LaTeX commands
            commands = self.latex_command_pattern.findall(text)
            self.learned_commands.update(commands)
            
            # Extract symbols
            symbols = self.symbol_pattern.findall(text)
            self.learned_symbols.update(symbols)
        
        # Filter by frequency
        frequent_commands = {
            cmd for cmd, count in self.learned_commands.items() 
            if count >= min_frequency
        }
        
        frequent_symbols = {
            sym for sym, count in self.learned_symbols.items()
            if count >= min_frequency
        }
        
        return {
            'commands': frequent_commands,
            'symbols': frequent_symbols,
            'operators': self._identify_operators(texts)
        }
    
    def _identify_operators(self, texts: List[str]) -> Set[str]:
        """Identify mathematical operators from context."""
        operators = set()
        operator_contexts = [
            (r'\b(\w+)\s*\(.*?\)', 'function'),  # Functions like sin(x), log(x)
            (r'\\(\w+)\s*\{', 'command_with_arg'),  # Commands like \frac{
            (r'([+\-*/^_=])', 'arithmetic'),  # Arithmetic operators
        ]
        
        for text in texts:
            for pattern, op_type in operator_contexts:
                matches = re.findall(pattern, text)
                operators.update(matches)
        
        return operators
    
    def is_latex_command(self, token: str) -> bool:
        """Check if token is a LaTeX command based on learned patterns."""
        if token.startswith('\\'):
            cmd = token[1:]
            return cmd in self.learned_commands or self._looks_like_latex(cmd)
        return False
    
    def _looks_like_latex(self, cmd: str) -> bool:
        """Heuristic: does this look like a LaTeX command?"""
        # LaTeX commands are typically lowercase or mixed case, alphabetic
        return cmd.isalpha() and (cmd.islower() or any(c.isupper() for c in cmd))


class DynamicCodeAnalyzer:
    """
    Dynamically identifies programming language patterns without hardcoded keywords.
    Uses statistical analysis and POS-like patterns.
    """
    
    def __init__(self):
        self.learned_keywords = Counter()
        self.learned_identifiers = Counter()
        self.control_flow_patterns = set()
        self.definition_patterns = set()
        
    def analyze_corpus(self, code_samples: List[str], language: Optional[str] = None) -> Dict[str, Set[str]]:
        """
        Analyze code corpus to learn keywords and patterns.
        
        Args:
            code_samples: List of code snippets
            language: Optional language hint for better analysis
            
        Returns:
            Dictionary with 'keywords', 'builtins', 'control_flow', 'definitions'
        """
        for code in code_samples:
            self._analyze_code_structure(code)
        
        # Classify tokens by usage patterns
        keywords = self._identify_keywords()
        builtins = self._identify_builtins(code_samples)
        
        return {
            'keywords': keywords,
            'builtins': builtins,
            'control_flow': self.control_flow_patterns,
            'definitions': self.definition_patterns
        }
    
    def _analyze_code_structure(self, code: str):
        """Analyze code structure to identify patterns."""
        lines = code.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            
            # Identify control flow patterns (if/for/while followed by condition)
            control_match = re.match(r'(\w+)\s*\(', line)
            if control_match:
                self.control_flow_patterns.add(control_match.group(1))
            
            # Identify definition patterns (def/function/class followed by name)
            def_match = re.match(r'(\w+)\s+(\w+)\s*[:\(]', line)
            if def_match:
                keyword, identifier = def_match.groups()
                self.definition_patterns.add(keyword)
                self.learned_identifiers[identifier] += 1
                
            # Count all word tokens
            words = re.findall(r'\b([a-zA-Z_]\w*)\b', line)
            self.learned_keywords.update(words)
    
    def _identify_keywords(self) -> Set[str]:
        """
        Identify keywords using statistical patterns.
        Keywords typically:
        - Appear frequently
        - Are short (2-8 chars)
        - Are all lowercase or have specific casing (e.g., True/False)
        - Appear at statement boundaries
        """
        keywords = set()
        
        for token, count in self.learned_keywords.items():
            # Heuristics for keyword identification
            if (
                count >= 3 and  # Appears multiple times
                2 <= len(token) <= 10 and  # Reasonable length
                (token.islower() or token in ['True', 'False', 'None', 'NULL']) and
                token not in self.learned_identifiers  # Not a variable name
            ):
                keywords.add(token)
        
        return keywords
    
    def _identify_builtins(self, code_samples: List[str]) -> Set[str]:
        """
        Identify built-in functions/types by pattern:
        - Called with parentheses
        - Often used without definition
        """
        builtins = set()
        
        for code in code_samples:
            # Find function calls
            calls = re.findall(r'\b([a-zA-Z_]\w*)\s*\(', code)
            builtins.update(calls)
        
        # Filter out user-defined functions (those that appear after def/function)
        user_defined = set()
        for code in code_samples:
            defs = re.findall(r'(?:def|function|func)\s+([a-zA-Z_]\w*)', code)
            user_defined.update(defs)
        
        return builtins - user_defined
    
    def is_keyword(self, token: str) -> bool:
        """Check if token is likely a keyword based on learned patterns."""
        return (
            token in self.learned_keywords and
            token in self.control_flow_patterns or
            token in self.definition_patterns or
            self._looks_like_keyword(token)
        )
    
    def _looks_like_keyword(self, token: str) -> bool:
        """Heuristic: does this look like a keyword?"""
        return (
            len(token) <= 8 and
            token.islower() and
            token.isalpha()
        )
    
    def is_literal(self, token: str) -> bool:
        """Check if token is a literal (string, number, boolean)."""
        # Number
        if re.match(r'^-?\d+\.?\d*$', token):
            return True
        
        # String
        if (token.startswith('"') and token.endswith('"')) or \
           (token.startswith("'") and token.endswith("'")):
            return True
        
        # Boolean/None (language-agnostic patterns)
        if token in ['true', 'false', 'null', 'nil', 'True', 'False', 'None', 'NULL']:
            return True
        
        return False


class SemanticTokenClassifier:
    """
    Classifies tokens based on semantic context using pattern matching.
    Works across different domains (NL/MATH/CODE) dynamically.
    """
    
    def __init__(self):
        self.token_contexts = defaultdict(Counter)
        
    def learn_from_context(self, tokens: List[str], contexts: List[str]):
        """
        Learn token classifications from their contexts.
        
        Args:
            tokens: List of tokens
            contexts: List of context labels for each token
        """
        for token, context in zip(tokens, contexts):
            self.token_contexts[token][context] += 1
    
    def classify_token(self, token: str) -> str:
        """
        Classify a token based on learned contexts.
        
        Returns:
            'keyword', 'operator', 'identifier', 'literal', 'command', 'symbol'
        """
        if token not in self.token_contexts:
            return self._classify_by_heuristic(token)
        
        # Get most common context
        most_common = self.token_contexts[token].most_common(1)[0][0]
        return most_common
    
    def _classify_by_heuristic(self, token: str) -> str:
        """Classify unknown token by heuristic rules."""
        # LaTeX command
        if token.startswith('\\'):
            return 'command'
        
        # Operator/symbol
        if re.match(r'^[+\-*/=<>!&|^%]+$', token):
            return 'operator'
        
        # Number
        if re.match(r'^-?\d+\.?\d*$', token):
            return 'literal'
        
        # String
        if (token.startswith('"') and token.endswith('"')) or \
           (token.startswith("'") and token.endswith("'")):
            return 'literal'
        
        # Likely identifier
        if re.match(r'^[a-zA-Z_]\w*$', token):
            return 'identifier'
        
        return 'unknown'


def create_dynamic_math_tokenizer(corpus_samples: List[str], min_frequency: int = 2):
    """
    Factory function to create a math tokenizer trained on corpus.
    
    Args:
        corpus_samples: Sample mathematical texts
        min_frequency: Minimum frequency for pattern recognition
        
    Returns:
        Configured DynamicMathAnalyzer
    """
    analyzer = DynamicMathAnalyzer()
    analyzer.analyze_corpus(corpus_samples, min_frequency)
    return analyzer


def create_dynamic_code_tokenizer(corpus_samples: List[str], language: Optional[str] = None):
    """
    Factory function to create a code tokenizer trained on corpus.
    
    Args:
        corpus_samples: Sample code snippets
        language: Optional language hint
        
    Returns:
        Configured DynamicCodeAnalyzer
    """
    analyzer = DynamicCodeAnalyzer()
    analyzer.analyze_corpus(corpus_samples, language)
    return analyzer
