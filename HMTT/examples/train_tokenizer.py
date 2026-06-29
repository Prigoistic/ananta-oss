"""
Example: Training HMTT Tokenizer

This script demonstrates how to train the HMTT tokenizer on a corpus.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HMTT.training.corpus_builder import CorpusBuilder
from HMTT.training.vocab_trainer import VocabTrainer
from HMTT.utils.logging import get_logger


def main():
    """Train HMTT tokenizer."""
    logger = get_logger("HMTT.Example.Train")
    
    # Sample documents with mixed content
    documents = [
        "The equation $E = mc^2$ relates energy and mass.",
        "Calculate the integral: $$\\int_0^1 x^2 dx = \\frac{1}{3}$$",
        "Here's Python code: ```python\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\n```",
        "The quadratic formula $x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$ solves $ax^2+bx+c=0$.",
        "Implement sorting: ```python\ndata = [3, 1, 4, 1, 5]\ndata.sort()\nprint(data)\n```",
    ]
    
    # Add more training examples
    for i in range(100):
        documents.append(f"Training example {i} with some text and $x_{i}$ variable.")
    
    logger.info(f"Training on {len(documents)} documents")
    
    # Build corpus
    logger.info("Building corpus...")
    builder = CorpusBuilder()
    corpus_path = "data/corpus.txt"
    builder.build_corpus(documents, corpus_path, verbose=True)
    
    # Train vocabulary
    logger.info("Training BPE vocabulary...")
    trainer = VocabTrainer(vocab_size=10000, min_frequency=2)
    
    try:
        trainer.train(
            corpus_paths=[corpus_path],
            output_path="models/hmtt_tokenizer.json",
            verbose=True
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Vocabulary size: {trainer.get_vocab_size()}")
        
        # Test the tokenizer
        test_text = "Compute $\\sum_{i=1}^n i$ using ```python sum(range(n+1))```"
        logger.info(f"\nTest text: {test_text}")
        
        ids = trainer.encode_test(test_text)
        logger.info(f"Token IDs: {ids}")
        
        decoded = trainer.decode_test(ids)
        logger.info(f"Decoded: {decoded}")
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install: pip install tokenizers")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
