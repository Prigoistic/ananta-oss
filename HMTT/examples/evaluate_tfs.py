"""
Example: Evaluating Tokenization Quality

This script demonstrates how to evaluate tokenization quality using TFS metric.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HMTT.evaluation.tfs_metric import TFSEvaluator, compare_tokenizers
from HMTT.preprocessing.math_tokenizer import MathTokenizer
from HMTT.preprocessing.code_tokenizer import CodeTokenizer
from HMTT.preprocessing.nl_tokenizer import NLTokenizer
from HMTT.utils.logging import get_logger


def simple_tokenizer(text: str) -> list:
    """Simple whitespace tokenizer for comparison."""
    return text.split()


def main():
    """Evaluate tokenization quality."""
    logger = get_logger("HMTT.Example.Evaluate")
    
    # Test texts
    test_texts = [
        "The equation $x^2 + y^2 = r^2$ defines a circle.",
        "Calculate $$\\sum_{i=1}^n i = \\frac{n(n+1)}{2}$$",
        "Python code: ```python def hello(): print('Hi')```",
        "The quadratic formula: $x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$",
    ]
    
    # Initialize tokenizers
    math_tokenizer = MathTokenizer()
    nl_tokenizer = NLTokenizer(use_unicode=False)
    
    evaluator = TFSEvaluator()
    
    logger.info("Comparing tokenization approaches\n")
    
    for i, text in enumerate(test_texts, 1):
        logger.info(f"{'='*60}")
        logger.info(f"Test {i}: {text}")
        logger.info(f"{'='*60}")
        
        # Simple whitespace tokenization
        simple_tokens = simple_tokenizer(text)
        
        # HMTT-style tokenization (NL only for simplicity)
        hmtt_tokens = nl_tokenizer.tokenize(text)
        
        # Compare
        results = compare_tokenizers(
            text,
            simple_tokens,
            hmtt_tokens,
            "Simple (whitespace)",
            "HMTT (NL)"
        )
        
        for name, metrics in results.items():
            logger.info(f"\n{name}:")
            logger.info(f"  TFS Score: {metrics.tfs_score:.4f}")
            logger.info(f"  Atomic Splits: {metrics.atomic_splits}")
            logger.info(f"  Inappropriate Merges: {metrics.inappropriate_merges}")
            logger.info(f"  Total Tokens: {metrics.total_tokens}")
        
        logger.info("")
    
    # Test math-specific tokenization
    logger.info(f"\n{'='*60}")
    logger.info("Math-specific tokenization test")
    logger.info(f"{'='*60}")
    
    math_text = r"\frac{1}{2} + \alpha^2 = x_i"
    logger.info(f"Text: {math_text}")
    
    math_tokens = math_tokenizer.tokenize(math_text)
    logger.info(f"Tokens: {math_tokens}")
    
    metrics = evaluator.compute_tfs(math_text, math_tokens)
    logger.info(f"\nTFS Score: {metrics.tfs_score:.4f}")
    logger.info(f"Atomic Splits: {metrics.atomic_splits}")
    
    # Test dataset evaluation
    logger.info(f"\n{'='*60}")
    logger.info("Dataset evaluation")
    logger.info(f"{'='*60}")
    
    dataset_texts = test_texts
    dataset_tokens = [nl_tokenizer.tokenize(t) for t in dataset_texts]
    
    dataset_metrics = evaluator.evaluate_dataset(dataset_texts, dataset_tokens)
    logger.info(f"\nDataset Results:")
    logger.info(f"  Average TFS Score: {dataset_metrics.tfs_score:.4f}")
    logger.info(f"  Total Atomic Splits: {dataset_metrics.atomic_splits}")
    logger.info(f"  Total Inappropriate Merges: {dataset_metrics.inappropriate_merges}")
    logger.info(f"  Total Tokens: {dataset_metrics.total_tokens}")
    
    return 0


if __name__ == "__main__":
    exit(main())
