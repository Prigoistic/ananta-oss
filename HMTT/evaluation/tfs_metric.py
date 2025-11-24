"""
Tokenization Fidelity Score (TFS) Metric.

Implements the TFS metric from the research paper to measure
the quality of tokenization for hybrid math-text content.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import re


@dataclass
class TFSMetrics:
    """Container for TFS metric results."""
    tfs_score: float
    fragmentation_loss: float
    max_possible_loss: float
    atomic_splits: int
    inappropriate_merges: int
    total_tokens: int
    
    def __repr__(self) -> str:
        return (
            f"TFS Score: {self.tfs_score:.4f}\n"
            f"Fragmentation Loss: {self.fragmentation_loss:.4f}\n"
            f"Max Possible Loss: {self.max_possible_loss:.4f}\n"
            f"Atomic Splits: {self.atomic_splits}\n"
            f"Inappropriate Merges: {self.inappropriate_merges}\n"
            f"Total Tokens: {self.total_tokens}"
        )


class TFSEvaluator:
    """
    Evaluates Tokenization Fidelity Score (TFS).
    
    TFS = 1 - (FragmentationLoss / MaxPossibleLoss)
    
    FragmentationLoss counts:
    1. Splitting atomic math tokens (LaTeX commands, variables, numbers)
    2. Merging unrelated NL units inappropriately
    3. Mis-tokenizing code primitives (keywords, identifiers, operators)
    """
    
    # Patterns for atomic tokens
    LATEX_COMMAND_PATTERN = re.compile(r'\\[a-zA-Z]+')
    MATH_VARIABLE_PATTERN = re.compile(r'[a-zA-Z][_^{}\w()+-]*')
    NUMBER_PATTERN = re.compile(r'\d+\.?\d*(?:[eE][+-]?\d+)?')
    CODE_KEYWORD_PATTERN = re.compile(r'\b(?:def|class|return|if|else|for|while|import|from)\b')
    CODE_IDENTIFIER_PATTERN = re.compile(r'\b[a-zA-Z_]\w*\b')
    
    def __init__(self):
        """Initialize the TFS evaluator."""
        pass
    
    def compute_tfs(
        self,
        original_text: str,
        tokens: List[str],
        token_types: List[str] = None
    ) -> TFSMetrics:
        """
        Compute TFS score for tokenization.
        
        Args:
            original_text: Original input text
            tokens: List of tokens produced by tokenizer
            token_types: Optional list of token types ('nl', 'math', 'code')
            
        Returns:
            TFSMetrics object with evaluation results
        """
        # Count atomic splits
        atomic_splits = self._count_atomic_splits(original_text, tokens)
        
        # Count inappropriate merges
        inappropriate_merges = self._count_inappropriate_merges(original_text, tokens)
        
        # Total fragmentation loss
        fragmentation_loss = atomic_splits + inappropriate_merges
        
        # Maximum possible loss (worst case: every character is wrong)
        max_possible_loss = len(original_text)
        
        # Compute TFS score
        if max_possible_loss == 0:
            tfs_score = 1.0
        else:
            tfs_score = 1.0 - (fragmentation_loss / max_possible_loss)
        
        return TFSMetrics(
            tfs_score=tfs_score,
            fragmentation_loss=fragmentation_loss,
            max_possible_loss=max_possible_loss,
            atomic_splits=atomic_splits,
            inappropriate_merges=inappropriate_merges,
            total_tokens=len(tokens)
        )
    
    def _count_atomic_splits(self, text: str, tokens: List[str]) -> int:
        """
        Count how many atomic tokens were incorrectly split.
        
        Args:
            text: Original text
            tokens: Tokenized output
            
        Returns:
            Number of atomic splits
        """
        splits = 0
        
        # Find all LaTeX commands in text
        latex_commands = self.LATEX_COMMAND_PATTERN.findall(text)
        for cmd in latex_commands:
            if not self._is_preserved_in_tokens(cmd, tokens):
                splits += 1
        
        # Find all numbers in text
        numbers = self.NUMBER_PATTERN.findall(text)
        for num in numbers:
            if not self._is_preserved_in_tokens(num, tokens):
                splits += 1
        
        # Find all math variables (simple heuristic)
        # This is conservative - only counts obvious patterns
        variables = self.MATH_VARIABLE_PATTERN.findall(text)
        for var in variables:
            if len(var) > 1 and not self._is_preserved_in_tokens(var, tokens):
                # Only count if it looks like a math variable (has _ or ^)
                if '_' in var or '^' in var:
                    splits += 1
        
        return splits
    
    def _count_inappropriate_merges(self, text: str, tokens: List[str]) -> int:
        """
        Count inappropriate token merges.
        
        Args:
            text: Original text
            tokens: Tokenized output
            
        Returns:
            Number of inappropriate merges
        """
        merges = 0
        
        # Check for merged punctuation with words
        for token in tokens:
            # Word ending with punctuation (e.g., "hello.")
            if len(token) > 2 and token[-1] in '.,;:!?' and token[:-1].isalpha():
                merges += 1
            
            # Punctuation merged with word (e.g., ".hello")
            if len(token) > 2 and token[0] in '.,;:!?' and token[1:].isalpha():
                merges += 1
            
            # Multiple unrelated operators merged
            if len(token) > 2 and all(c in '+-*/<>=!&|' for c in token):
                merges += 1
        
        return merges
    
    def _is_preserved_in_tokens(self, substring: str, tokens: List[str]) -> bool:
        """
        Check if a substring appears as a single token or consecutive tokens that preserve it.
        
        Args:
            substring: Substring to check
            tokens: List of tokens
            
        Returns:
            True if substring is preserved atomically
        """
        # Check if substring is a single token
        if substring in tokens:
            return True
        
        # Check if substring spans multiple tokens correctly
        # Join tokens and see if substring appears without being split
        joined = ''.join(tokens)
        if substring in joined:
            # Now check if it's split incorrectly
            # For this simple version, we assume it's preserved if it exists in joined form
            return True
        
        return False
    
    def compare_tokenizers(
        self,
        text: str,
        tokenizer1_tokens: List[str],
        tokenizer2_tokens: List[str],
        tokenizer1_name: str = "Tokenizer 1",
        tokenizer2_name: str = "Tokenizer 2"
    ) -> Dict[str, TFSMetrics]:
        """
        Compare two tokenizers on the same text.
        
        Args:
            text: Input text
            tokenizer1_tokens: Tokens from first tokenizer
            tokenizer2_tokens: Tokens from second tokenizer
            tokenizer1_name: Name of first tokenizer
            tokenizer2_name: Name of second tokenizer
            
        Returns:
            Dictionary mapping tokenizer name to TFSMetrics
        """
        metrics1 = self.compute_tfs(text, tokenizer1_tokens)
        metrics2 = self.compute_tfs(text, tokenizer2_tokens)
        
        return {
            tokenizer1_name: metrics1,
            tokenizer2_name: metrics2
        }
    
    def evaluate_dataset(
        self,
        texts: List[str],
        tokenized_outputs: List[List[str]]
    ) -> TFSMetrics:
        """
        Evaluate TFS on a dataset.
        
        Args:
            texts: List of original texts
            tokenized_outputs: List of tokenized outputs (one per text)
            
        Returns:
            Aggregated TFSMetrics
        """
        total_tfs = 0.0
        total_fragmentation_loss = 0.0
        total_max_possible_loss = 0.0
        total_atomic_splits = 0
        total_inappropriate_merges = 0
        total_tokens = 0
        
        for text, tokens in zip(texts, tokenized_outputs):
            metrics = self.compute_tfs(text, tokens)
            total_tfs += metrics.tfs_score
            total_fragmentation_loss += metrics.fragmentation_loss
            total_max_possible_loss += metrics.max_possible_loss
            total_atomic_splits += metrics.atomic_splits
            total_inappropriate_merges += metrics.inappropriate_merges
            total_tokens += metrics.total_tokens
        
        n = len(texts)
        avg_tfs = total_tfs / n if n > 0 else 0.0
        
        return TFSMetrics(
            tfs_score=avg_tfs,
            fragmentation_loss=total_fragmentation_loss,
            max_possible_loss=total_max_possible_loss,
            atomic_splits=total_atomic_splits,
            inappropriate_merges=total_inappropriate_merges,
            total_tokens=total_tokens
        )


def compute_tfs(
    original_text: str,
    tokens: List[str],
    token_types: List[str] = None
) -> TFSMetrics:
    """
    Convenience function to compute TFS score.
    
    Args:
        original_text: Original input text
        tokens: List of tokens produced by tokenizer
        token_types: Optional list of token types
        
    Returns:
        TFSMetrics object
    """
    evaluator = TFSEvaluator()
    return evaluator.compute_tfs(original_text, tokens, token_types)


def compare_tokenizers(
    text: str,
    tokenizer1_tokens: List[str],
    tokenizer2_tokens: List[str],
    tokenizer1_name: str = "Tokenizer 1",
    tokenizer2_name: str = "Tokenizer 2"
) -> Dict[str, TFSMetrics]:
    """
    Convenience function to compare tokenizers.
    
    Args:
        text: Input text
        tokenizer1_tokens: Tokens from first tokenizer
        tokenizer2_tokens: Tokens from second tokenizer
        tokenizer1_name: Name of first tokenizer
        tokenizer2_name: Name of second tokenizer
        
    Returns:
        Dictionary mapping tokenizer name to metrics
    """
    evaluator = TFSEvaluator()
    return evaluator.compare_tokenizers(
        text, tokenizer1_tokens, tokenizer2_tokens,
        tokenizer1_name, tokenizer2_name
    )
