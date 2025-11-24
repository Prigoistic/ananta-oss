#!/usr/bin/env python3
"""
HMTT Command-Line Interface.

Provides commands for training, encoding, decoding, and evaluating HMTT tokenizers.
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HMTT.training.corpus_builder import CorpusBuilder
from HMTT.training.vocab_trainer import VocabTrainer
from HMTT.inference.encoder import HMTTEncoder
from HMTT.inference.decoder import HMTTDecoder
from HMTT.evaluation.tfs_metric import TFSEvaluator
from HMTT.utils.logging import get_logger
from HMTT.utils.io import read_text_file, write_text_file, load_jsonl


def cmd_build_corpus(args):
    """Build corpus from documents."""
    logger = get_logger("HMTT.CLI.BuildCorpus")
    
    builder = CorpusBuilder(
        code_language=args.code_language,
        use_unicode_nl=not args.no_unicode
    )
    
    if args.jsonl:
        count = builder.build_corpus_from_jsonl(
            args.input,
            args.output,
            text_field=args.text_field,
            verbose=args.verbose
        )
    elif args.text_files:
        count = builder.build_corpus_from_files(
            args.input.split(','),
            args.output,
            verbose=args.verbose
        )
    else:
        logger.error("Specify --jsonl or --text-files")
        return 1
    
    logger.info(f"Built corpus with {count} documents")
    return 0


def cmd_train_vocab(args):
    """Train BPE vocabulary."""
    logger = get_logger("HMTT.CLI.TrainVocab")
    
    trainer = VocabTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency
    )
    
    corpus_files = args.corpus.split(',')
    
    try:
        trainer.train(
            corpus_paths=corpus_files,
            output_path=args.output,
            verbose=args.verbose
        )
        
        logger.info(f"Trained vocabulary: {trainer.get_vocab_size()} tokens")
        
        if args.save_vocab_json:
            trainer.save_vocab_to_json(args.save_vocab_json)
            logger.info(f"Saved vocabulary to {args.save_vocab_json}")
        
        return 0
    
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install tokenizers")
        return 1


def cmd_encode(args):
    """Encode text to token IDs."""
    logger = get_logger("HMTT.CLI.Encode")
    
    try:
        encoder = HMTTEncoder(args.tokenizer)
        if args.input == '-':
            text = sys.stdin.read()
        else:
            text = read_text_file(args.input)
        
        ids = encoder.encode(text, add_special_tokens=not args.no_special_tokens)
        
        if args.output:
            write_text_file(' '.join(map(str, ids)), args.output)
        else:
            print(' '.join(map(str, ids)))
        
        if args.verbose:
            logger.info(f"Encoded {len(text)} chars -> {len(ids)} tokens")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


def cmd_decode(args):
    """Decode token IDs to text."""
    logger = get_logger("HMTT.CLI.Decode")
    
    try:
        decoder = HMTTDecoder(args.tokenizer)
        
        if args.input == '-':
            ids_str = sys.stdin.read()
        else:
            ids_str = read_text_file(args.input)
        
        ids = list(map(int, ids_str.split()))
        text = decoder.decode(ids, skip_special_tokens=not args.keep_special_tokens)
        
        if args.output:
            write_text_file(text, args.output)
        else:
            print(text)
        
        if args.verbose:
            logger.info(f"Decoded {len(ids)} tokens -> {len(text)} chars")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


def cmd_evaluate(args):
    """Evaluate tokenization quality."""
    logger = get_logger("HMTT.CLI.Evaluate")
    
    try:
        encoder = HMTTEncoder(args.tokenizer)
        evaluator = TFSEvaluator()
        
        # Load test data
        if args.jsonl:
            data = load_jsonl(args.input)
            texts = [item[args.text_field] for item in data]
        else:
            text = read_text_file(args.input)
            texts = [text]
        
        # Evaluate
        all_metrics = []
        
        for text in texts:
            tokens = encoder.tokenize_text(text)
            metrics = evaluator.compute_tfs(text, tokens)
            all_metrics.append(metrics)
        
        # Aggregate
        avg_tfs = sum(m.tfs_score for m in all_metrics) / len(all_metrics)
        total_splits = sum(m.atomic_splits for m in all_metrics)
        total_merges = sum(m.inappropriate_merges for m in all_metrics)
        total_tokens = sum(m.total_tokens for m in all_metrics)
        
        logger.info(f"Evaluated {len(texts)} documents")
        logger.info(f"Average TFS Score: {avg_tfs:.4f}")
        logger.info(f"Total Atomic Splits: {total_splits}")
        logger.info(f"Total Inappropriate Merges: {total_merges}")
        logger.info(f"Total Tokens: {total_tokens}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="HMTT: Hybrid Math-Text Tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Build corpus
    parser_build = subparsers.add_parser('build-corpus', help='Build training corpus')
    parser_build.add_argument('input', help='Input file or directory')
    parser_build.add_argument('output', help='Output corpus file')
    parser_build.add_argument('--jsonl', action='store_true', help='Input is JSONL')
    parser_build.add_argument('--text-files', action='store_true', help='Input is comma-separated text files')
    parser_build.add_argument('--text-field', default='text', help='JSONL text field')
    parser_build.add_argument('--code-language', default='python', help='Code language')
    parser_build.add_argument('--no-unicode', action='store_true', help='Disable unicode NL tokenization')
    parser_build.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Train vocabulary
    parser_train = subparsers.add_parser('train-vocab', help='Train BPE vocabulary')
    parser_train.add_argument('corpus', help='Corpus file(s) (comma-separated)')
    parser_train.add_argument('output', help='Output tokenizer file')
    parser_train.add_argument('--vocab-size', type=int, default=50000, help='Vocabulary size')
    parser_train.add_argument('--min-frequency', type=int, default=2, help='Minimum token frequency')
    parser_train.add_argument('--save-vocab-json', help='Save vocabulary as JSON')
    parser_train.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Encode
    parser_encode = subparsers.add_parser('encode', help='Encode text to token IDs')
    parser_encode.add_argument('tokenizer', help='Tokenizer file')
    parser_encode.add_argument('input', help='Input file (- for stdin)')
    parser_encode.add_argument('--output', help='Output file')
    parser_encode.add_argument('--no-special-tokens', action='store_true', help='Skip special tokens')
    parser_encode.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Decode
    parser_decode = subparsers.add_parser('decode', help='Decode token IDs to text')
    parser_decode.add_argument('tokenizer', help='Tokenizer file')
    parser_decode.add_argument('input', help='Input file (- for stdin)')
    parser_decode.add_argument('--output', help='Output file')
    parser_decode.add_argument('--keep-special-tokens', action='store_true', help='Keep special tokens')
    parser_decode.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Evaluate
    parser_eval = subparsers.add_parser('evaluate', help='Evaluate tokenization quality')
    parser_eval.add_argument('tokenizer', help='Tokenizer file')
    parser_eval.add_argument('input', help='Input file')
    parser_eval.add_argument('--jsonl', action='store_true', help='Input is JSONL')
    parser_eval.add_argument('--text-field', default='text', help='JSONL text field')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to command
    commands = {
        'build-corpus': cmd_build_corpus,
        'train-vocab': cmd_train_vocab,
        'encode': cmd_encode,
        'decode': cmd_decode,
        'evaluate': cmd_evaluate,
    }
    
    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
