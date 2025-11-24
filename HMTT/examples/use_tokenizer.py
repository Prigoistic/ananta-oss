"""
Example: Using HMTT Encoder/Decoder

This script demonstrates how to use a trained HMTT tokenizer for encoding and decoding.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HMTT.inference.encoder import HMTTEncoder
from HMTT.inference.decoder import HMTTDecoder
from HMTT.evaluation.tfs_metric import compute_tfs
from HMTT.utils.logging import get_logger


def main():
    """Demonstrate HMTT encoding and decoding."""
    logger = get_logger("HMTT.Example.Inference")
    
    # Path to trained tokenizer
    tokenizer_path = "models/hmtt_tokenizer.json"
    
    if not Path(tokenizer_path).exists():
        logger.error(f"Tokenizer not found: {tokenizer_path}")
        logger.error("Please run train_tokenizer.py first")
        return 1
    
    try:
        # Initialize encoder and decoder
        logger.info("Loading tokenizer...")
        encoder = HMTTEncoder(tokenizer_path)
        decoder = HMTTDecoder(tokenizer_path)
        
        logger.info(f"Vocabulary size: {encoder.get_vocab_size()}")
        
        # Test texts
        test_texts = [
            "The Pythagorean theorem: $a^2 + b^2 = c^2$",
            "Calculate: $$\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}$$",
            "Python function: ```python\ndef greet(name):\n    return f'Hello, {name}!'\n```",
            "Einstein's equation $E = mc^2$ where $c$ is the speed of light.",
        ]
        
        for i, text in enumerate(test_texts, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Test {i}")
            logger.info(f"{'='*60}")
            logger.info(f"Original: {text}")
            
            # Encode
            ids = encoder.encode(text)
            logger.info(f"Encoded: {len(ids)} tokens")
            logger.info(f"Token IDs: {ids[:20]}..." if len(ids) > 20 else f"Token IDs: {ids}")
            
            # Get token strings
            tokens = encoder.tokenize_text(text)
            logger.info(f"Tokens: {tokens}")
            
            # Decode
            decoded = decoder.decode(ids)
            logger.info(f"Decoded: {decoded}")
            
            # Evaluate
            metrics = compute_tfs(text, tokens)
            logger.info(f"\nTFS Metrics:")
            logger.info(f"  Score: {metrics.tfs_score:.4f}")
            logger.info(f"  Atomic Splits: {metrics.atomic_splits}")
            logger.info(f"  Inappropriate Merges: {metrics.inappropriate_merges}")
        
        logger.info("\n" + "="*60)
        logger.info("All tests completed!")
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install: pip install tokenizers")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
