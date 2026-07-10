# Ananta

A domain-aware tokenizer and LoRA fine-tuning pipeline for mathematical reasoning in language models. An energy-based self-correction mechanism and a symbolic verification subsystem are designed and described in the project's research papers, but neither is implemented in this repository.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![Development Status](https://img.shields.io/badge/status-research%20%2F%20active%20development-orange)]()

## Overview

Standard subword tokenizers (the BPE and byte-level BPE schemes used in GPT-4, LLaMA, and most open-weight models) treat mathematical notation as an arbitrary character sequence, because they were trained almost entirely on natural-language and code corpora. This produces systematic fragmentation: `\frac` is split into `[\, fr, ac]`, `x_{i+1}` is cut at the brace boundary, and an unfamiliar LaTeX command is decomposed into subword shards with no individual meaning. When the atomic units of symbolic reasoning (operators, LaTeX commands, subscripted variables) are split across token boundaries, gradient updates from a derivation error get distributed across meaningless character shards instead of the operation that actually caused the error. Ananta addresses this failure mode with HMTT, a domain-aware pre-tokenization stage that partitions text into natural-language, math, and code regions before BPE merges are learned, so these atomic units are never split.

The rest of the repository is a LoRA fine-tuning and evaluation pipeline that fine-tunes `deepseek-ai/deepseek-math-7b` on the DeepMind Mathematics Dataset. Two other components referenced in the project's design documents and research papers, an Energy-Based Self-Learning Engine (EBT-SLE) and a Recursive Logic Subsystem (RLS), are architectural proposals only. See [Architecture](#architecture) and [Status and limitations](#status-and-limitations) for what is and isn't present in the code.

## Architecture

### HMTT: Hybrid Math-Text Tokenizer

HMTT solves the fragmentation problem by inserting a domain-aware pre-tokenization stage before any BPE merges are applied.

**Stage 1: region partitioning.** [`TextPartitioner`](HMTT/preprocessing/partitioner.py) uses regex to detect LaTeX delimiters (`$...$`, `$$...$$`, `\[...\]`, `\begin{equation}`/`align`/`eqnarray`/`gather`/`multline`) and code fences (`` `...` ``, ` ```...``` `). Every character in the input is assigned to one of three region types, `nl`, `math`, or `code`, with start and end offsets, and the remaining text is filled in as `nl`. This step matters because it guarantees the domain-specific tokenizers below never operate across a modality boundary.

**Stage 2: domain-specific tokenizers.** Each region is handed to a dedicated tokenizer:

- [`MathTokenizer`](HMTT/preprocessing/math_tokenizer.py) walks LaTeX character by character and extracts commands (`\frac`, `\alpha`, `\sum`), numbers, braced groups (with nested-brace tracking via `_extract_braced_content`), and composite variables with subscripts/superscripts (`x_i`, `\theta^{(t)}`) as atomic pre-tokens that later BPE merges cannot split. The command set is not hardcoded: `DynamicMathAnalyzer.analyze_corpus` in [`dynamic_analyzer.py`](HMTT/preprocessing/dynamic_analyzer.py) counts `\command` occurrences across a corpus and keeps those above `min_frequency`. Calling `tokenizer.tokenize(text, learn=True)` also lets the tokenizer add unrecognized commands to its set on the fly, one document at a time.
- [`CodeTokenizer`](HMTT/preprocessing/code_tokenizer.py) is structured to prefer tree-sitter AST tokenization and fall back to a regex scanner. In the current code, `_init_tree_sitter` always sets `tree_sitter_available = False` regardless of whether the `tree_sitter` package is installed (the class comment marks this as a placeholder pending grammar integration), so every call goes through `_tokenize_with_regex` today. `DynamicCodeAnalyzer` identifies keywords by statistical heuristics (short, lowercase, frequent, appearing in control-flow or definition position) rather than a per-language keyword list.
- [`NLTokenizer`](HMTT/preprocessing/nl_tokenizer.py) applies a GPT-4-style regex pattern for contractions, words, numbers, and punctuation, with an optional Unicode-aware variant via the `regex` package and an ASCII-only fallback.

**Stage 3: constrained BPE training.** [`CorpusBuilder`](HMTT/training/corpus_builder.py) partitions each document, tokenizes each region with its domain tokenizer, and writes the concatenated pre-token stream (JSONL, plain text, or streamed input are all supported). [`VocabTrainer`](HMTT/training/vocab_trainer.py) feeds this pre-tokenized stream into HuggingFace `tokenizers`' `BpeTrainer` with `WhitespaceSplit` as the pre-tokenizer, so BPE only ever merges within a pre-token, never across one. `VocabTrainer` also declares its own fixed `LATEX_COMMANDS` and `CODE_KEYWORDS` lists as reserved vocabulary entries; this hardcoded list coexists with, and is separate from, the dynamically learned command set used during pre-tokenization. The documented default vocabulary size is 256,000 tokens, but the CLI's `--vocab-size` flag defaults to 50,000. The two disagree, so set it explicitly when training.

**Stage 4: encoding and decoding.** [`HMTTEncoder`](HMTT/inference/encoder.py) re-runs the stage 1 and 2 pre-tokenization, applies the trained BPE model, and returns token IDs, optionally with character offsets. [`HMTTDecoder`](HMTT/inference/decoder.py) reverses this and applies light regex cleanup (spacing around punctuation and brackets) on the reconstructed text.

**Stage 5: quality measurement.** [`TFSEvaluator`](HMTT/evaluation/tfs_metric.py) computes the Tokenization Fidelity Score:

```
TFS = 1 - (FragmentationLoss / MaxPossibleLoss)
```

`FragmentationLoss` counts atomic units (LaTeX commands, numbers, subscripted variables) that were split across multiple tokens, plus a small set of heuristic "inappropriate merge" patterns (punctuation fused to a word, runs of operators fused together). `MaxPossibleLoss` is simply the character length of the input, so TFS is a normalized, per-document score in `(-∞, 1]` that's comparable across tokenizers on the same text via `compare_tokenizers`.

All five stages are exposed through a CLI:

```bash
python -m HMTT build-corpus --input docs/ --output corpus.txt
python -m HMTT train-vocab corpus.txt tokenizer.json --vocab-size 256000
python -m HMTT encode tokenizer.json - <<< "Solve \frac{x^2}{2} = 1"
python -m HMTT evaluate tokenizer.json eval.jsonl --jsonl
```

The HMTT design and the TFS metric are described in Ghosh et al., "Bridging the Semantic Gap: A Hybrid Math-Text Tokenizer for Enhanced Logical Reasoning in Large Language Models" ([ResearchGate](https://www.researchgate.net/publication/393773297/Bridging_the_Semantic_Gap_A_Hybrid_Math-Text_Tokenizer_for_Enhanced_Logical_Reasoning_in_Large_Language_Models)).

### Recursive Logic Subsystem (RLS): not implemented

The RLS is proposed as a symbolic verifier that would check generated derivations step by step against a formal logic or computer-algebra backend, feeding a pass/fail (or graded) verification signal back into generation or training. The repository contains an empty placeholder directory, `RSL/`, with no source files. No autoformalization, SMT/ATP solver integration, or lemma-discovery code exists anywhere in the codebase. The proposed design and a critical assessment of its feasibility are described in Ghosh et al., "A Critical Analysis of the Proposed Recursive Logic Subsystem for Self-Learning LLMs in Scientific Discovery" ([ResearchGate](https://www.researchgate.net/publication/395473790/A_Critical_Analysis_of_the_Proposed_Recursive_Logic_Subsystem_for_Self-Learning_LLMs_in_Scientific_Discovery)).

### Energy-Based Self-Learning Engine (EBT-SLE): not implemented

The EBT-SLE is proposed as a mechanism that would replace or augment next-token likelihood training with an energy functional over full reasoning traces, minimized via contrastive-divergence-style sampling and shaped by the RLS's verification signal. The repository contains an empty placeholder directory, `EBSL-Engine/`, with no source files: no energy functional, no sampler, and no verifier-guided gradient computation exist in the code. The proposed formulation is described in Ghosh et al., "An Energy-Based Self-Learning Engine for Neuro-Symbolic Scientific Reasoning" ([ResearchGate](https://www.researchgate.net/publication/397885858/An_Energy-Based_Self-Learning_Engine_for_Neuro-Symbolic_Scientific_Reasoning)).

### The proposed full loop

The diagram below is the intended architecture connecting all three components, as described across the papers above. Only the generation and fine-tuning path (solid) is implemented. The verification and energy-minimization path (dashed) is not.

```
                 +--------------------+
   text  ----->  |   HMTT (encode)    |  — implemented
                 +---------+----------+
                           |
                           v
                 +--------------------+
                 | DeepSeek-Math-7B   |
                 |   + LoRA adapter   |  — implemented (src/training/)
                 +---------+----------+
                           |
                           v
                 +--------------------+
                 |  generated trace   |
                 +---------+----------+
                           :
                           v
                 +--------------------+
                 |   RLS verifier     |  — proposed only (RSL/ is empty)
                 +---------+----------+
                           :
                           v
                 +--------------------+
                 |  EBT-SLE energy    |  — proposed only (EBSL-Engine/ is empty)
                 |    minimization    |
                 +---------+----------+
                           :
                           +....> feedback into generation (proposed, not implemented)
```

The overview paper connecting these three pieces into a single system proposal is Ghosh et al., "Ananta: A Self-Learning LLM for Symbolic and Scientific Reasoning" ([ResearchGate](https://www.researchgate.net/publication/392438202/Ananta_A_Self-Learning_LLM_for_Symbolic_and_Scientific_Reasoning)).

## Fine-tuning and evaluation pipeline

The training system fine-tunes `deepseek-ai/deepseek-math-7b` with LoRA on the [DeepMind Mathematics Dataset](https://github.com/deepmind/mathematics_dataset), which ships problem-solution pairs at five difficulty levels (`train-easy`, `train-medium`, `train-hard`, `interpolate`, `extrapolate`). Three processors in [`src/data/`](src/data/) handle different layouts of that dataset: [`MathDatasetProcessor`](src/data/data_processor.py) expects the standard DeepMind directory structure, [`FlexibleMathDatasetProcessor`](src/data/flexible_data_processor.py) auto-detects alternative directory-naming conventions, and [`simple_data_converter.py`](src/data/simple_data_converter.py) is a minimal plaintext-to-JSON converter. All three emit instruction/input/output records.

Two training entry points are provided:

- [`easy_train.py`](src/training/easy_train.py) is a single-file pipeline sized for RTX 3050-class hardware: 8-bit quantized model load, LoRA (`r=8`, `alpha=32`, `dropout=0.1`) on `q_proj`/`v_proj`, 512-token max sequence length, a 90/10 train/validation split, and checkpoints every 500 steps.
- [`train_ananta.py`](src/training/train_ananta.py) wraps `AnantaTrainer` around TRL's `SFTTrainer`, reads hyperparameters from a JSON config (default: [`configs/train_config.json`](configs/train_config.json), learning rate `5e-5`, sequence length 1024, cosine schedule with `adamw_torch`), supports optional Weights & Biases logging, and writes final metrics to `training_metrics.json`.

[`AnantaEvaluator`](src/evaluation/evaluate_model.py) generates responses for held-out problems, extracts a numerical answer via regex (`extract_numerical_answer`), compares it to the reference answer within a floating-point tolerance, and reports accuracy broken down by difficulty level, response-time statistics, and basic error-pattern analysis. [`run_pipeline.py`](src/run_pipeline.py) chains data processing, training, evaluation, and Hugging Face Spaces deployment prep (`--step all`) into a single command by shelling out to the underlying scripts.

## Results

No benchmark numbers, evaluation logs, or a `results/`-style directory currently exist in this repository. `AnantaEvaluator` and the TFS metric are functional evaluation tools (running them against a trained checkpoint or a tokenized corpus produces `evaluation_summary.json` and `TFSMetrics` output respectively), but no such run has been checked into the repo. No accuracy or TFS numbers should be assumed here until someone reproduces them. To generate real numbers:

```bash
# TFS score for a tokenizer on a JSONL dataset
python -m HMTT evaluate tokenizer.json eval.jsonl --jsonl

# Accuracy by difficulty level for a fine-tuned checkpoint
python src/evaluation/evaluate_model.py --model_path ./deepseek_finetuned \
    --dataset_path formatted_math_dataset.json
```

Both commands require the DeepMind Mathematics Dataset, and the second also requires a fine-tuned checkpoint. Neither is bundled with this repository.

## Installation and quickstart

```bash
git clone https://github.com/Prigoistic/ananta-oss.git
cd ananta-oss/ananta-update
pip install -r requirements.txt
pip install -e HMTT/
```

```python
from HMTT.preprocessing import MathTokenizer
from HMTT.preprocessing.partitioner import TextPartitioner

# Domain-aware LaTeX tokenization with atomicity preserved
tokenizer = MathTokenizer(min_frequency=2)
print(tokenizer.tokenize(r"\frac{x^2}{2} + \alpha \cdot x"))
# ['\\frac', '{x^2}', '{2}', '+', '\\alpha', '\\cdot', 'x']

# Region partitioning across NL / math / code
regions = TextPartitioner().partition("The integral $\\int_0^1 f(x)\\,dx$ is bounded.")
for r in regions:
    print(r.type, repr(r.text))
```

## Project structure

```
ananta-update/
├── HMTT/                            # Hybrid Math-Text Tokenizer
│   ├── preprocessing/
│   │   ├── partitioner.py           # Splits text into nl/math/code regions
│   │   ├── math_tokenizer.py        # Atomic LaTeX tokenizer with dynamic command learning
│   │   ├── code_tokenizer.py        # Regex code tokenizer (tree-sitter path is a stub)
│   │   ├── nl_tokenizer.py          # GPT-4-style natural language tokenizer
│   │   └── dynamic_analyzer.py      # Statistical keyword/command discovery
│   ├── training/
│   │   ├── corpus_builder.py        # Documents to partitioned, tokenized corpus
│   │   └── vocab_trainer.py         # BPE training with atomic pre-tokens
│   ├── inference/
│   │   ├── encoder.py               # Text to token IDs, with offset tracking
│   │   └── decoder.py               # Token IDs to text
│   ├── evaluation/
│   │   └── tfs_metric.py            # Tokenization Fidelity Score
│   ├── examples/                    # Runnable usage demos
│   ├── cli.py                       # `python -m HMTT ...` entry point
│   └── utils/                       # I/O and logging helpers
├── RSL/                             # Recursive Logic Subsystem, empty placeholder
├── EBSL-Engine/                     # Energy-Based Self-Learning Engine, empty placeholder
├── src/
│   ├── data/
│   │   ├── data_processor.py        # Standard DeepMind dataset processor
│   │   ├── flexible_data_processor.py  # Auto-detecting variant
│   │   └── simple_data_converter.py    # Plaintext to JSON converter
│   ├── training/
│   │   ├── easy_train.py            # Single-file LoRA pipeline (RTX 3050 class)
│   │   └── train_ananta.py          # AnantaTrainer wrapping TRL's SFTTrainer
│   ├── evaluation/
│   │   └── evaluate_model.py        # AnantaEvaluator: per-difficulty accuracy and timing
│   └── run_pipeline.py              # Orchestrates data, train, evaluate, deploy
├── configs/
│   └── train_config.json            # Default training hyperparameters
├── demos/
│   └── app.py                       # Gradio interface (works without a trained model)
├── deployment/
│   └── huggingface/deploy_hf_spaces.py  # HF Spaces packaging
├── tests/
│   ├── test_hmtt.py                 # HMTT unit tests (pytest)
│   └── test_ananta.py               # Manual interactive model-inference script
└── docs/                            # Quick-start guide and contributing notes
```

## Papers

- Ghosh, P. et al. "Ananta: A Self-Learning LLM for Symbolic and Scientific Reasoning." [ResearchGate](https://www.researchgate.net/publication/392438202/Ananta_A_Self-Learning_LLM_for_Symbolic_and_Scientific_Reasoning)
- Ghosh, P. et al. "Bridging the Semantic Gap: A Hybrid Math-Text Tokenizer for Enhanced Logical Reasoning in Large Language Models." [ResearchGate](https://www.researchgate.net/publication/393773297/Bridging_the_Semantic_Gap_A_Hybrid_Math-Text_Tokenizer_for_Enhanced_Logical_Reasoning_in_Large_Language_Models)
- Ghosh, P. et al. "A Critical Analysis of the Proposed Recursive Logic Subsystem for Self-Learning LLMs in Scientific Discovery." [ResearchGate](https://www.researchgate.net/publication/395473790/A_Critical_Analysis_of_the_Proposed_Recursive_Logic_Subsystem_for_Self-Learning_LLMs_in_Scientific_Discovery)
- Ghosh, P. et al. "An Energy-Based Self-Learning Engine for Neuro-Symbolic Scientific Reasoning." [ResearchGate](https://www.researchgate.net/publication/397885858/An_Energy-Based_Self-Learning_Engine_for_Neuro-Symbolic_Scientific_Reasoning)

Each paper corresponds to a component discussed under [Architecture](#architecture): the overview paper describes the full proposed system, and the remaining three cover HMTT, RLS, and EBT-SLE respectively.

## Citation

```bibtex
@article{ghosh2025ananta,
  title   = {Ananta: A Self-Learning LLM for Symbolic and Scientific Reasoning},
  author  = {Ghosh, Priyam and others},
  year    = {2025},
  url     = {https://www.researchgate.net/publication/392438202/Ananta_A_Self-Learning_LLM_for_Symbolic_and_Scientific_Reasoning}
}
```

## Status and limitations

**HMTT tokenizer.** Implemented and unit-tested (`tests/test_hmtt.py`): partitioning, the three domain tokenizers, dynamic command/keyword learning, constrained BPE training, encoding/decoding, and the TFS metric all run end to end. The code tokenizer's tree-sitter path is a stub that currently always falls through to the regex tokenizer regardless of whether `tree-sitter` is installed. `VocabTrainer`'s hardcoded `LATEX_COMMANDS`/`CODE_KEYWORDS` reserved-token lists sit alongside the dynamically learned command set from `DynamicMathAnalyzer`, so "no hardcoded lists" (as stated in the preprocessing docstrings) is true for pre-tokenization but not for vocabulary reservation. The CLI's default vocab size (50,000) and the trainer class's documented default (256,000) disagree, so pass `--vocab-size` explicitly.

**Data processing and LoRA fine-tuning.** Implemented. `easy_train.py` and `train_ananta.py` both run against `deepseek-ai/deepseek-math-7b` and require a Hugging Face token with access to that model. `src/data/data_processor.py:35` hardcodes a Windows path (`C:\Users\r0b0t1x\Desktop\...`) as its dataset default. Pass `--dataset_dir` explicitly or use `flexible_data_processor.py`, which doesn't have this issue.

**Evaluation and demo.** `AnantaEvaluator` and the Gradio app in `demos/app.py` are implemented and run without a trained model present (the demo falls back to dataset-only mode). Neither has been run against a real checkpoint in this repository, so no accuracy numbers are checked in. See [Results](#results).

**RLS and EBT-SLE.** Not implemented. `RSL/` and `EBSL-Engine/` are empty directories. The symbolic verifier, autoformalization/solver integration, energy functional, and contrastive-divergence training described in the corresponding papers have no source code in this repository.

**License file.** `setup.py` and `HMTT/setup.py` both declare an MIT classifier, but no `LICENSE` file is present in the repository at this time.

This is research code under active development, not a packaged release. The HMTT tokenizer is the one component that can be installed and used independently today. The fine-tuning pipeline requires GPU hardware, a Hugging Face token, and the DeepMind Mathematics Dataset to run. RLS and EBT-SLE are design proposals backed by their respective papers, not runnable code.

## License

`setup.py` declares MIT, but no `LICENSE` file currently exists in this repository. Treat the licensing status as unresolved until one is added.

**Project lead:** Priyam Ghosh, [github.com/Prigoistic/ananta-oss](https://github.com/Prigoistic/ananta-oss)
