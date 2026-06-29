"""
Vocabulary Trainer for HMTT.

Trains a BPE vocabulary using HuggingFace tokenizers with constraints
to preserve atomicity of math, code, and number tokens.
"""

from typing import List, Optional, Set, Dict
from pathlib import Path
import json

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from tokenizers.normalizers import Sequence as NormalizerSequence
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False


class VocabTrainer:
    """
    Trains a BPE vocabulary for HMTT.
    
    Enforces constraints:
    - Math commands remain atomic (\\frac, \\alpha, etc.)
    - Variable patterns remain atomic (x_i, theta^{(t)}, etc.)
    - Code keywords remain atomic
    - Numbers remain atomic
    - No merging across LaTeX commands, identifiers, or numbers
    """
    
    # Special tokens
    SPECIAL_TOKENS = [
        "<pad>",
        "<s>",
        "</s>",
        "<unk>",
        "<mask>",
    ]
    
    # Common LaTeX commands to reserve
    LATEX_COMMANDS = [
        "\\frac", "\\sqrt", "\\sum", "\\prod", "\\int", "\\lim",
        "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\theta",
        "\\pi", "\\sigma", "\\omega", "\\Omega",
        "\\mathbb", "\\mathbf", "\\mathcal", "\\text",
        "\\left", "\\right", "\\cdot", "\\times",
    ]
    
    # Common code keywords to reserve
    CODE_KEYWORDS = [
        "def", "class", "return", "if", "else", "elif", "for", "while",
        "import", "from", "as", "with", "try", "except", "finally",
        "True", "False", "None", "null", "undefined",
        "function", "const", "let", "var", "async", "await",
    ]
    
    def __init__(
        self,
        vocab_size: int = 256000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
        reserved_tokens: Optional[List[str]] = None
    ):
        """
        Initialize the vocabulary trainer.
        
        Args:
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for a token to be included
            special_tokens: Custom special tokens (default: SPECIAL_TOKENS)
            reserved_tokens: Additional reserved tokens
        """
        if not TOKENIZERS_AVAILABLE:
            raise ImportError(
                "HuggingFace tokenizers not available. "
                "Install with: pip install tokenizers"
            )
        
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Build special tokens list
        self.special_tokens = special_tokens or self.SPECIAL_TOKENS
        
        # Build reserved tokens list
        self.reserved_tokens = (reserved_tokens or []) + \
                               self.LATEX_COMMANDS + \
                               self.CODE_KEYWORDS
        
        # Initialize tokenizer
        self.tokenizer = None
    
    def train(
        self,
        corpus_paths: List[str],
        output_path: str,
        verbose: bool = False
    ):
        """
        Train BPE vocabulary on corpus files.
        
        Args:
            corpus_paths: List of corpus file paths
            output_path: Path to save trained tokenizer
            verbose: Print training progress
        """
        # Initialize BPE model
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        
        # Pre-tokenizer (split on whitespace - corpus is already pre-tokenized)
        self.tokenizer.pre_tokenizer = WhitespaceSplit()
        
        # Create trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=verbose,
            initial_alphabet=self._get_initial_alphabet(),
        )
        
        # Train on corpus files
        if verbose:
            print(f"Training BPE vocabulary on {len(corpus_paths)} files...")
        
        self.tokenizer.train(files=corpus_paths, trainer=trainer)
        
        # Save tokenizer
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(output_file))
        
        if verbose:
            print(f"Tokenizer saved to {output_path}")
            print(f"Vocabulary size: {self.tokenizer.get_vocab_size()}")
    
    def train_from_iterator(
        self,
        corpus_iterator: List[str],
        output_path: str,
        verbose: bool = False
    ):
        """
        Train BPE vocabulary from an iterator of text lines.
        
        Args:
            corpus_iterator: Iterator of pre-tokenized text lines
            output_path: Path to save trained tokenizer
            verbose: Print training progress
        """
        # Initialize BPE model
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = WhitespaceSplit()
        
        # Create trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=verbose,
            initial_alphabet=self._get_initial_alphabet(),
        )
        
        # Train on iterator
        if verbose:
            print("Training BPE vocabulary from iterator...")
        
        self.tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)
        
        # Save tokenizer
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(output_file))
        
        if verbose:
            print(f"Tokenizer saved to {output_path}")
            print(f"Vocabulary size: {self.tokenizer.get_vocab_size()}")
    
    def _get_initial_alphabet(self) -> List[str]:
        """
        Get initial alphabet for BPE training.
        
        Returns:
            List of initial tokens (characters and reserved tokens)
        """
        # ASCII characters
        alphabet = [chr(i) for i in range(32, 127)]
        
        # Common unicode characters
        alphabet.extend([
            # Greek letters
            'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ',
            'ν', 'ξ', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
            'Γ', 'Δ', 'Θ', 'Λ', 'Ξ', 'Π', 'Σ', 'Φ', 'Ψ', 'Ω',
        ])
        
        # Add reserved tokens
        alphabet.extend(self.reserved_tokens)
        
        return alphabet
    
    def add_reserved_tokens(self, tokens: List[str]):
        """
        Add additional reserved tokens.
        
        Args:
            tokens: List of tokens to reserve
        """
        self.reserved_tokens.extend(tokens)
    
    def load_tokenizer(self, path: str):
        """
        Load a trained tokenizer.
        
        Args:
            path: Path to tokenizer file
        """
        self.tokenizer = Tokenizer.from_file(path)
    
    def get_vocab(self) -> Dict[str, int]:
        """
        Get the vocabulary mapping.
        
        Returns:
            Dictionary of token -> ID
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        
        return self.tokenizer.get_vocab()
    
    def get_vocab_size(self) -> int:
        """
        Get vocabulary size.
        
        Returns:
            Size of vocabulary
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        
        return self.tokenizer.get_vocab_size()
    
    def encode_test(self, text: str) -> List[int]:
        """
        Test encode text with trained tokenizer.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        
        encoding = self.tokenizer.encode(text)
        return encoding.ids
    
    def decode_test(self, ids: List[int]) -> str:
        """
        Test decode token IDs with trained tokenizer.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        
        return self.tokenizer.decode(ids)
    
    def save_vocab_to_json(self, output_path: str):
        """
        Save vocabulary to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        
        vocab = self.get_vocab()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
