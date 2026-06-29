# Ananta

A domain-aware tokenizer and LoRA fine-tuning pipeline for mathematical reasoning in language models, with a planned energy-based self-correction mechanism that is not yet implemented.

---

## The Problem

Standard BPE tokenizers—the kind used in GPT-4, LLaMA, and most open-weight models—treat mathematical notation as arbitrary character sequences. This produces systematic fragmentation: `\frac` becomes `[\, fr, ac]`, the subscript `x_{i+1}` is split at the brace boundary, and a LaTeX command your model has never seen gets decomposed into meaningless subword shards. The damage is not cosmetic. When the atomic units of symbolic reasoning—variables, operators, LaTeX commands—are split across token boundaries, the model must learn that `fr` and `ac` together mean "fraction." That is wasted capacity, and it degrades chain-of-thought stability for arithmetic derivations because the structural signal is buried under a tokenization artifact.

- ✅ Solving PhD-level mathematics and physics problems.
- ✅ Generating fully correct, stepwise derivations
- ✅ Discovering new lemmas and mathematical relations
- ✅ Autoformalizing scientific papers into formal logic
- ✅ Creating new scientific hypotheses grounded in symbolic verification

**Core Philosophy:** Go beyond typical transformer-based language models by constructing a hybrid architecture that understands symbolic math deeply, produces correct reasoning steps, verifies its own reasoning using logic, reduces hallucination, learns recursively, and evolves toward self-improving AI.

---

## The Approach

### HMTT: Hybrid Math-Text Tokenizer

The core implemented component is HMTT, which solves the fragmentation problem by inserting a domain-aware pre-tokenization stage before BPE merges are applied.

**Stage 1 — Region partitioning** (`HMTT/preprocessing/partitioner.py`)

Text is segmented into typed regions before any tokenization occurs. The partitioner uses regex to detect LaTeX delimiters (`$...$`, `$$...$$`, `\[...\]`, `\begin{equation}...\end{equation}`) and code fences (`` `...` ``, ```` ```...``` ````). Every character in the input is assigned to one of three region types—`NL`, `MATH`, or `CODE`—with start and end offsets. This segmentation is the load-bearing step: it ensures that subsequent tokenizers never operate across modality boundaries.

**Stage 2 — Domain-specific tokenizers**

Each region type is passed to a dedicated tokenizer:

`MathTokenizer` (`math_tokenizer.py`) extracts LaTeX commands (`\frac`, `\alpha`, `\sum`), composite variables with subscripts and superscripts (`x_i`, `\theta^{(t)}`), numbers, and operators as atomic pre-tokens. No merge can split these units later. The key design choice is that the command list is not hardcoded. Instead, `DynamicMathAnalyzer` (`dynamic_analyzer.py`) learns which `\command` strings appear above a minimum frequency threshold in the training corpus, adding them to the protected set automatically. You can call `tokenizer.tokenize(text, learn=True)` during corpus ingestion to build the command inventory on the fly.

`CodeTokenizer` (`code_tokenizer.py`) uses tree-sitter for AST-based tokenization when a grammar is available, with a regex fallback. `DynamicCodeAnalyzer` identifies keywords statistically—tokens that appear in syntactically significant positions across the corpus are elevated to keyword status, making the tokenizer language-agnostic. It has been tested against Python and arbitrary DSLs.

`NLTokenizer` (`nl_tokenizer.py`) applies a GPT-4-style regex pattern, handling contractions and optionally using the `regex` module for full Unicode coverage, with an ASCII-safe fallback.

**Stage 3 — Constrained BPE training** (`HMTT/training/vocab_trainer.py`)

Pre-tokens from all three stages are fed into HuggingFace's `tokenizers` library for BPE vocabulary training. The trainer is configured to treat atomic pre-tokens as reserved units that BPE is not permitted to split or merge across. The default vocabulary size is 256k. The training data is built by `CorpusBuilder` (`corpus_builder.py`), which processes JSONL or plain-text documents in batches, partitions each document, applies the three domain tokenizers, and writes a stream of pre-token sequences.

**Stage 4 — Encoding and decoding** (`HMTT/inference/`)

`HMTTEncoder` encodes text to token IDs with offset tracking for character-level alignment. `HMTTDecoder` reconstructs text with special-token filtering and post-processing cleanup.

**Stage 5 — Quality measurement** (`HMTT/evaluation/tfs_metric.py`)

The Tokenization Fidelity Score is defined as:

```
TFS = 1 - (FragmentationLoss / MaxPossibleLoss)
```

where `FragmentationLoss` counts atomic splits (a protected unit that was split) and inappropriate merges. The metric produces a `TFSMetrics` dataclass with a per-category breakdown and supports comparison between tokenizers on the same dataset.

A CLI wraps all five stages:

```bash
python -m HMTT build-corpus --input docs/ --output corpus.txt
python -m HMTT train-vocab --corpus corpus.txt --vocab-size 256000
python -m HMTT encode --model tokenizer.json --text "Solve \frac{x^2}{2} = 1"
python -m HMTT evaluate --model tokenizer.json --dataset eval.jsonl
```

---

### Fine-Tuning Pipeline

The training system fine-tunes `deepseek-ai/deepseek-math-7b` with LoRA on the DeepMind Mathematics Dataset. The dataset contains problem-solution pairs at five difficulty levels: `train-easy`, `train-medium`, `train-hard`, `interpolate`, and `extrapolate`. Three data processors handle the various file layouts the dataset ships in: `MathDatasetProcessor` (standard DeepMind format), `FlexibleDataProcessor` (auto-detects structure), and `SimpleDataConverter` (plaintext to JSON). All three produce instruction-input-output tuples.

Two training scripts are provided:

`easy_train.py` is a single-file pipeline designed for RTX 3050 class hardware. It loads `formatted_math_dataset.json`, applies an 8-bit NF8 quantization config, attaches LoRA adapters (`r=8`, `alpha=32`, `dropout=0.1`) to the `q_proj` and `v_proj` projections, tokenizes sequences to 512 tokens maximum, and runs a 90/10 train-validation split. Checkpoints are saved every 500 steps.

`train_ananta.py` wraps `AnantaTrainer` around TRL's `SFTTrainer`. It reads a JSON config file, supports WandB integration, streams structured logs to a file, saves evaluation metrics to JSON at the end of each epoch, and supports checkpoint resumption. The default config sets learning rate to `5e-5`, sequence length to 1024, and uses a cosine learning rate schedule with `adamw_torch`.

`AnantaEvaluator` (`src/evaluation/evaluate_model.py`) runs the fine-tuned model over held-out problems, extracts numerical answers with regex patterns, compares them to ground truth with a floating-point tolerance, and reports accuracy broken down by difficulty level along with response timing and error-pattern statistics.

---

### Planned: EB-SLE and RLS

The existing README describes an Energy-Based Self-Learning Engine (EB-SLE) that would replace next-token prediction with energy minimization over reasoning traces, and a Recursive Logic Subsystem (RLS) that would verify generated derivations symbolically. Neither is implemented in the current codebase. Directories `RLS/` and `EBSL-Engine/` exist as placeholders. The theoretical architecture described in the original documentation—energy functionals, contrastive divergence sampling, verifier-guided gradients, SMT/ATP solver integration, lemma discovery—is not present in any source file.

---

## Why It Works

The HMTT approach is grounded in a concrete hypothesis: token boundary alignment with symbolic structure improves the signal-to-noise ratio of the training objective for mathematical text. When `\frac` is a single token, gradient updates from derivation errors can directly target the fraction operation. When it is three tokens, the same error is distributed across character shards that have no individual semantic meaning.

The statistical approach to command discovery rather than a hardcoded keyword list has a practical payoff: the tokenizer generalizes to any LaTeX package, any custom `\newcommand`, and any DSL without modification to the source.

The connection to energy-based models, Hopfield networks, or attractor dynamics that appear in the project's design documents is not grounded in the current implementation and is not made here.

---

## Quickstart

```bash
git clone https://github.com/Prigoistic/ananta-oss.git
cd ananta-oss/ananta-update
pip install -r requirements.txt
pip install -e HMTT/
```

```python
from HMTT.preprocessing import MathTokenizer

tokenizer = MathTokenizer(min_frequency=2)
tokens = tokenizer.tokenize(r"\frac{x^2}{2} + \alpha \cdot x", learn=True)
print(tokens)
# ['\frac', '{', 'x', '^', '2', '}', '/', '2', '}', '+', '\alpha', '\cdot', 'x']
```

```python
from HMTT.preprocessing.partitioner import Partitioner

p = Partitioner()
regions = p.partition("The integral $\\int_0^1 f(x)\\,dx$ is bounded.")
for r in regions:
    print(r.type, repr(r.text))
# NL 'The integral '
# MATH '\\int_0^1 f(x)\\,dx'
# NL ' is bounded.'
```

---

## Project Structure

```
ananta-update/
├── HMTT/                           # Hybrid Math-Text Tokenizer (complete)
│   ├── preprocessing/
│   │   ├── partitioner.py          # Splits text into NL/MATH/CODE regions
│   │   ├── math_tokenizer.py       # LaTeX tokenizer with atomicity preservation
│   │   ├── code_tokenizer.py       # Language-agnostic code tokenizer
│   │   ├── nl_tokenizer.py         # GPT-4-style natural language tokenizer
│   │   └── dynamic_analyzer.py     # Corpus-based pattern learning engine
│   ├── training/
│   │   ├── corpus_builder.py       # Streams documents into pre-token sequences
│   │   └── vocab_trainer.py        # BPE training with atomicity constraints
│   ├── inference/
│   │   ├── encoder.py              # Text → token IDs with offset tracking
│   │   └── decoder.py              # Token IDs → text
│   ├── evaluation/
│   │   └── tfs_metric.py           # Tokenization Fidelity Score
│   ├── examples/                   # Runnable usage demos
│   ├── cli.py                      # Command-line interface
│   └── utils/                      # I/O and logging helpers
├── src/
│   ├── data/
│   │   ├── data_processor.py       # DeepMind dataset processor
│   │   ├── flexible_data_processor.py  # Auto-detecting dataset processor
│   │   └── simple_data_converter.py    # Plain-text to JSON converter
│   ├── training/
│   │   ├── easy_train.py           # Single-file LoRA pipeline (RTX 3050)
│   │   └── train_ananta.py         # AnantaTrainer with full config support
│   ├── evaluation/
│   │   └── evaluate_model.py       # AnantaEvaluator with per-difficulty metrics
│   └── run_pipeline.py             # Orchestrates data → train → evaluate
├── configs/
│   └── train_config.json           # Default training hyperparameters
├── demos/
│   └── app.py                      # Gradio web interface
├── deployment/
│   └── deploy_hf_spaces.py         # HuggingFace Spaces packaging
├── tests/
│   ├── test_hmtt.py                # HMTT unit tests
│   └── test_ananta.py              # Model loading and inference tests
└── docs/                           # Additional documentation
```

---

## Status

**HMTT tokenizer** — Complete. The full pipeline from text partitioning through BPE training, encoding, decoding, and TFS evaluation is implemented and tested. The dynamic learning components work as documented.

**Data processing** — Complete. All three processors handle the DeepMind dataset format and produce correctly structured training files.

**LoRA fine-tuning** — Complete. Both `easy_train.py` and `train_ananta.py` are functional. The base model path defaults to `deepseek-ai/deepseek-math-7b` and requires a HuggingFace token for download.

**Evaluation** — Complete. `AnantaEvaluator` runs end-to-end and produces per-difficulty accuracy breakdowns.

**Demo interface** — Complete. The Gradio app in `demos/app.py` handles cases where neither the dataset nor the fine-tuned model is present.

**EB-SLE (Energy-Based Self-Learning Engine)** — Not implemented. The concept is documented and the directory exists, but there is no source code. The energy functional, contrastive divergence training, and verifier-guided gradient updates described in early design documents have no corresponding implementation.

**RLS (Recursive Logic Subsystem)** — Not implemented. The symbolic verifier, autoformalization engine, and lemma discovery system are planned but absent from the codebase.

Several internal paths in `data_processor.py` hard-code a Windows filepath (`C:\Users\r0b0t1x\Desktop\...`). These should be updated or replaced with an environment variable before the processor will work without modification on other machines.

---

## License

MIT. See `LICENSE`.

---

**Project lead:** Priyam Ghosh — [github.com/Prigoistic/ananta-oss](https://github.com/Prigoistic/ananta-oss)
