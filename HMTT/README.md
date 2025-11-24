# Hybrid Math-Text Tokenizer (HMTT)

## Overview

HMTT is a **discrete tokenization system** that combines natural language, mathematical LaTeX, and code tokenization with BPE (Byte-Pair Encoding) vocabulary learning. It is **NOT** an encoder, VAE, or latent vector model—it is purely symbolic and BPE-based.

## Key Features

- **Domain-Aware Partitioning**: Automatically segments text into Natural Language, Math (LaTeX), and Code regions
- **Structure-Aware Tokenization**: 
  - Math: Preserves LaTeX commands, variables, and numbers atomically
  - Code: AST-aware tokenization of identifiers, keywords, and operators
  - NL: GPT-4 style regex-based tokenization
- **BPE Vocabulary Training**: Unified vocabulary with atomicity constraints
- **Tokenization Fidelity Score (TFS)**: Evaluation metric for tokenization quality

## Architecture

```
HMTT/
├── preprocessing/          # Text partitioning and domain tokenizers
│   ├── partitioner.py      # NL/MATH/CODE segmentation
│   ├── math_tokenizer.py   # LaTeX-aware math tokenizer
│   ├── code_tokenizer.py   # AST-based code tokenizer
│   └── nl_tokenizer.py     # GPT-4 style NL tokenizer
├── training/               # Corpus building and vocabulary training
│   ├── corpus_builder.py   # Pre-tokenization pipeline
│   └── vocab_trainer.py    # BPE vocabulary trainer
├── inference/              # Encoding and decoding
│   ├── encoder.py          # Text → Token IDs
│   └── decoder.py          # Token IDs → Text
├── evaluation/             # Quality metrics
│   └── tfs_metric.py       # Tokenization Fidelity Score
└── utils/                  # Utilities
    ├── io.py               # File I/O helpers
    └── logging.py          # Logging utilities
```

## Installation

### Requirements

```bash
# Core dependencies
pip install tokenizers>=0.13.0

# Optional dependencies
pip install regex  # For unicode-aware NL tokenization
pip install tree-sitter  # For AST-based code tokenization
pip install pytest  # For running tests
```

### Install HMTT

```bash
cd ananta/
pip install -e .
```

## Quick Start

### 1. Training a Tokenizer

```python
from HMTT.training.corpus_builder import CorpusBuilder
from HMTT.training.vocab_trainer import VocabTrainer

# Prepare documents
documents = [
    "The equation $E = mc^2$ relates energy and mass.",
    "Calculate: $$\\int_0^1 x^2 dx = \\frac{1}{3}$$",
    "Python code: ```python\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\n```"
]

# Build corpus
builder = CorpusBuilder()
builder.build_corpus(documents, "corpus.txt")

# Train vocabulary
trainer = VocabTrainer(vocab_size=50000)
trainer.train(["corpus.txt"], "tokenizer.json", verbose=True)
```

### 2. Encoding Text

```python
from HMTT.inference.encoder import HMTTEncoder

encoder = HMTTEncoder("tokenizer.json")

text = "The formula $x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$ solves quadratic equations."
token_ids = encoder.encode(text)
tokens = encoder.tokenize_text(text)

print(f"Token IDs: {token_ids}")
print(f"Tokens: {tokens}")
```

### 3. Decoding Tokens

```python
from HMTT.inference.decoder import HMTTDecoder

decoder = HMTTDecoder("tokenizer.json")

decoded_text = decoder.decode(token_ids)
print(f"Decoded: {decoded_text}")
```

### 4. Evaluating Quality

```python
from HMTT.evaluation.tfs_metric import compute_tfs

metrics = compute_tfs(original_text, tokens)
print(f"TFS Score: {metrics.tfs_score:.4f}")
print(f"Atomic Splits: {metrics.atomic_splits}")
print(f"Inappropriate Merges: {metrics.inappropriate_merges}")
```

## Pipeline Details

### Stage 1: Partitioning

The `TextPartitioner` segments input text into regions:

- **Math**: `$...$`, `$$...$$`, `\[...\]`, `\begin{equation}...\end{equation}`
- **Code**: `` `...` ``, ` ```...``` `
- **Natural Language**: Everything else

### Stage 2: Domain-Specific Tokenization

Each region is tokenized according to its domain:

#### Math Tokenizer
- LaTeX commands: `\frac`, `\alpha`, `\sum`
- Variables: `x_i`, `\theta^{(t)}`
- Numbers: `3.14159`
- Operators: `+`, `-`, `\cdot`
- Preserves atomicity of commands and variables

#### Code Tokenizer
- Identifiers: `factorial`, `my_var`
- Keywords: `def`, `class`, `return`
- Literals: `"hello"`, `123`
- Operators: `+`, `==`, `->`
- Falls back to regex if tree-sitter unavailable

#### NL Tokenizer
- GPT-4 style regex patterns
- Handles contractions: `don't`, `won't`
- Preserves numbers atomically
- Splits punctuation appropriately

### Stage 3: BPE Training

The `VocabTrainer` enforces constraints:
- LaTeX commands remain atomic
- Variable patterns remain atomic
- Code keywords remain atomic
- Numbers remain atomic
- No merging across domain boundaries

### Stage 4: Encoding/Decoding

- **Encoder**: Applies pre-tokenization → BPE merges → token IDs
- **Decoder**: Converts token IDs → text with formatting preservation

## Evaluation: TFS Metric

The **Tokenization Fidelity Score (TFS)** measures tokenization quality:

```
TFS = 1 - (FragmentationLoss / MaxPossibleLoss)
```

Where `FragmentationLoss` counts:
1. Splitting atomic math tokens
2. Merging unrelated NL units
3. Mis-tokenizing code primitives

### Example

```python
from HMTT.evaluation.tfs_metric import TFSEvaluator

evaluator = TFSEvaluator()

text = r"The command \alpha appears in $x_i + \alpha$"
tokens = ["The", "command", r"\alpha", "appears", "in", "x_i", "+", r"\alpha"]

metrics = evaluator.compute_tfs(text, tokens)
print(metrics)
```

## Examples

See the `examples/` directory:
- `train_tokenizer.py`: Full training pipeline
- `use_tokenizer.py`: Encoding/decoding examples
- `evaluate_tfs.py`: Quality evaluation

Run examples:
```bash
cd HMTT/examples
python train_tokenizer.py
python use_tokenizer.py
python evaluate_tfs.py
```

## Testing

Run the test suite:
```bash
cd ananta/
pytest tests/test_hmtt.py -v
```

## Design Principles

1. **Discrete, Not Continuous**: HMTT is a symbolic tokenizer, not a neural encoder
2. **Atomicity First**: Never split formal symbols (math commands, variables, numbers)
3. **Domain Awareness**: Different tokenization strategies for NL, Math, and Code
4. **BPE-Based**: Uses standard BPE with constraints for vocabulary learning
5. **Lossless**: Decoding should perfectly reconstruct the original text

## Limitations

- Tree-sitter support is optional (falls back to regex for code)
- Unicode support requires `regex` module (falls back to ASCII patterns)
- LaTeX command list is extensible but not exhaustive
- Code language support is primarily Python (extensible)

## Citation

This implementation is based on the research paper:
"Bridging the Semantic Gap: A Hybrid Math-Text Tokenizer"

## License

Part of the Ananta project.

## Contributing

See `CONTRIBUTING.md` for guidelines.
