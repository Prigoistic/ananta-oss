"""
Math Tokenizer for HMTT.

Structure-aware LaTeX tokenizer that extracts atomic mathematical units
without rendering or encoding - purely symbolic splitting.
"""

import re
from typing import List, Set


class MathTokenizer:
    """
    Structure-aware LaTeX tokenizer for mathematical expressions.
    
    Extracts atomic units:
    - LaTeX commands (\\frac, \\sum, \\alpha, etc.)
    - Operators (+, -, \\cdot, etc.)
    - Numbers (123, 3.14, etc.)
    - Variables with subscripts/superscripts (x_i, \\theta^{(t)})
    - Braced groups (only when syntactically correct)
    - Parentheses and brackets
    
    Maintains atomicity - never splits:
    - LaTeX commands
    - Variable structures
    - Numbers
    """
    
    # Common LaTeX commands (extendable)
    LATEX_COMMANDS = {
        # Greek letters
        'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
        'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'pi', 'rho', 'sigma',
        'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega',
        'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta',
        'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Xi', 'Pi', 'Rho', 'Sigma',
        'Tau', 'Upsilon', 'Phi', 'Chi', 'Psi', 'Omega',
        
        # Math operators
        'frac', 'sqrt', 'sum', 'prod', 'int', 'lim', 'inf', 'sup',
        'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
        'arcsin', 'arccos', 'arctan',
        'sinh', 'cosh', 'tanh',
        'log', 'ln', 'exp',
        'min', 'max', 'arg', 'det', 'dim', 'ker',
        
        # Arrows and relations
        'rightarrow', 'leftarrow', 'Rightarrow', 'Leftarrow', 'leftrightarrow',
        'to', 'mapsto', 'implies', 'iff',
        'leq', 'geq', 'neq', 'approx', 'equiv', 'sim', 'propto',
        'subset', 'subseteq', 'supset', 'supseteq',
        'in', 'notin', 'cup', 'cap', 'emptyset',
        
        # Delimiters
        'left', 'right', 'big', 'Big', 'bigg', 'Bigg',
        
        # Formatting
        'mathbb', 'mathbf', 'mathcal', 'mathit', 'mathrm', 'mathsf',
        'text', 'textbf', 'textit',
        'over', 'choose',
        
        # Spacing
        'quad', 'qquad', 'space', 'hspace', 'vspace',
        
        # Special
        'partial', 'nabla', 'infty', 'cdot', 'cdots', 'ldots', 'ddots',
        'times', 'div', 'pm', 'mp',
    }
    
    def __init__(self):
        """Initialize the math tokenizer."""
        # Build command pattern
        command_list = '|'.join(sorted(self.LATEX_COMMANDS, key=len, reverse=True))
        self.command_pattern = re.compile(
            r'\\(?:' + command_list + r')\b'
        )
    
    def tokenize(self, math_text: str) -> List[str]:
        """
        Tokenize mathematical LaTeX expression into atomic units.
        
        Args:
            math_text: LaTeX mathematical expression
            
        Returns:
            List of atomic token strings
        """
        tokens = []
        i = 0
        
        while i < len(math_text):
            # Skip whitespace
            if math_text[i].isspace():
                i += 1
                continue
            
            # LaTeX command
            if math_text[i] == '\\':
                cmd_match = self.command_pattern.match(math_text, i)
                if cmd_match:
                    tokens.append(cmd_match.group(0))
                    i = cmd_match.end()
                    continue
                else:
                    # Single character escape like \{, \}, etc.
                    if i + 1 < len(math_text):
                        tokens.append(math_text[i:i+2])
                        i += 2
                    else:
                        tokens.append(math_text[i])
                        i += 1
                    continue
            
            # Numbers (including decimals and scientific notation)
            if math_text[i].isdigit():
                num_match = re.match(r'\d+\.?\d*(?:[eE][+-]?\d+)?', math_text[i:])
                if num_match:
                    tokens.append(num_match.group(0))
                    i += num_match.end()
                    continue
            
            # Variable with subscript/superscript
            if math_text[i].isalpha():
                var_token = math_text[i]
                i += 1
                
                # Check for subscript or superscript
                while i < len(math_text) and math_text[i] in '_^':
                    var_token += math_text[i]
                    i += 1
                    
                    # Handle braced subscript/superscript
                    if i < len(math_text) and math_text[i] == '{':
                        brace_content, end_idx = self._extract_braced_content(math_text, i)
                        var_token += brace_content
                        i = end_idx
                    # Handle single character subscript/superscript
                    elif i < len(math_text) and (math_text[i].isalnum() or math_text[i] in '+-'):
                        var_token += math_text[i]
                        i += 1
                
                tokens.append(var_token)
                continue
            
            # Braced groups {content}
            if math_text[i] == '{':
                brace_content, end_idx = self._extract_braced_content(math_text, i)
                tokens.append(brace_content)
                i = end_idx
                continue
            
            # Parentheses and brackets (individual tokens)
            if math_text[i] in '()[]':
                tokens.append(math_text[i])
                i += 1
                continue
            
            # Multi-character operators
            if i + 1 < len(math_text):
                two_char = math_text[i:i+2]
                if two_char in ['==', '!=', '<=', '>=', '<<', '>>', '**', '||', '&&']:
                    tokens.append(two_char)
                    i += 2
                    continue
            
            # Single character operators and symbols
            if math_text[i] in '+-*/=<>!&|~^,%:;':
                tokens.append(math_text[i])
                i += 1
                continue
            
            # Any other character
            tokens.append(math_text[i])
            i += 1
        
        return tokens
    
    def _extract_braced_content(self, text: str, start: int) -> tuple[str, int]:
        """
        Extract content within braces, handling nested braces.
        
        Args:
            text: Full text
            start: Index of opening brace
            
        Returns:
            Tuple of (braced content including braces, index after closing brace)
        """
        if text[start] != '{':
            return '', start
        
        depth = 0
        i = start
        
        while i < len(text):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1], i + 1
            elif text[i] == '\\' and i + 1 < len(text):
                # Skip escaped characters
                i += 1
            i += 1
        
        # Unclosed brace
        return text[start:], len(text)
    
    def is_atomic(self, token: str) -> bool:
        """
        Check if a token should be treated as atomic (never split).
        
        Args:
            token: Token string
            
        Returns:
            True if token is atomic
        """
        # LaTeX commands are atomic
        if token.startswith('\\'):
            return True
        
        # Numbers are atomic
        if re.match(r'^\d+\.?\d*(?:[eE][+-]?\d+)?$', token):
            return True
        
        # Variables with subscripts/superscripts are atomic
        if re.match(r'^[a-zA-Z][_^{}\w()+-]*$', token):
            return True
        
        return False
