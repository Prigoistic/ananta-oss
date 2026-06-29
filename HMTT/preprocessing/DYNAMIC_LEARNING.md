# Dynamic Tokenization with Learning

HMTT now uses **dynamic content analysis** instead of hardcoded keyword lists. The tokenizers learn patterns from your corpus automatically.

## How It Works

### 1. Math Tokenizer - Learns LaTeX Commands

```python
from HMTT.preprocessing import MathTokenizer

# Option 1: Learn from corpus
corpus = [
    r"\frac{x^2}{2} + \alpha",
    r"\int_0^1 f(x) dx",
    r"\sum_{i=1}^n x_i"
]
tokenizer = MathTokenizer(corpus_samples=corpus)

# Option 2: Learn on-the-fly
tokenizer = MathTokenizer()  # Empty initially
tokens = tokenizer.tokenize(r"\newcommand{test}")  # Learns \newcommand automatically
```

### 2. Code Tokenizer - Learns Keywords

```python
from HMTT.preprocessing import CodeTokenizer

# Option 1: Learn from corpus
corpus = [
    "def hello(): return True",
    "class MyClass: pass",
    "for i in range(10): print(i)"
]
tokenizer = CodeTokenizer(language="python", corpus_samples=corpus)

# Option 2: Learn on-the-fly
tokenizer = CodeTokenizer(language="python")
tokens = tokenizer.tokenize("async def main(): await task()")  # Learns 'async', 'await'
```

## Dynamic Analysis Features

### Math Analyzer (`DynamicMathAnalyzer`)

- **Extracts LaTeX commands**: Finds all `\command` patterns
- **Identifies symbols**: Mathematical operators and special symbols
- **Pattern learning**: Learns from frequency and context
- **No hardcoding**: Pure pattern-based extraction

```python
from HMTT.preprocessing.dynamic_analyzer import DynamicMathAnalyzer

analyzer = DynamicMathAnalyzer()
patterns = analyzer.analyze_corpus([
    r"\customop{x} + \myfunction{y}",
    r"\customop{z} - 5"
])
print(patterns['commands'])  # {'customop', 'myfunction'}
```

### Code Analyzer (`DynamicCodeAnalyzer`)

- **Statistical keyword detection**: Identifies keywords by usage patterns
- **Control flow recognition**: Finds if/for/while/def patterns
- **Built-in detection**: Distinguishes builtins from user functions
- **Language-agnostic**: Works with any programming language

```python
from HMTT.preprocessing.dynamic_analyzer import DynamicCodeAnalyzer

analyzer = DynamicCodeAnalyzer()
patterns = analyzer.analyze_corpus([
    "fnc greet(name): print(name)",
    "fnc main(): greet('World')"
])
print(patterns['keywords'])  # {'fnc'} - learned from structure!
```

### Semantic Classifier (`SemanticTokenClassifier`)

- **Context-based classification**: Classifies tokens by usage
- **Multi-domain**: Works across NL/MATH/CODE
- **Learning from context**: Improves with more examples

```python
from HMTT.preprocessing.dynamic_analyzer import SemanticTokenClassifier

classifier = SemanticTokenClassifier()
classifier.learn_from_context(
    tokens=['def', 'hello', '+', '42'],
    contexts=['keyword', 'identifier', 'operator', 'literal']
)
print(classifier.classify_token('def'))  # 'keyword'
```

## Key Advantages

### ✅ No Hardcoding
- Zero hardcoded keyword lists
- No predefined command dictionaries
- Learns everything from data

### ✅ Language-Agnostic
- Works with any programming language
- Supports custom DSLs
- Handles domain-specific notation

### ✅ Adaptive Learning
- Improves with more data
- Learns new patterns on-the-fly
- Adapts to coding styles

### ✅ Corpus-Based
- Train once on your codebase
- Learns your conventions
- Preserves project-specific patterns

## Advanced Usage

### Pre-train on Large Corpus

```python
from HMTT.preprocessing import MathTokenizer, CodeTokenizer

# Load your corpus
math_corpus = load_math_papers()  # Your function
code_corpus = load_code_files()   # Your function

# Train tokenizers
math_tok = MathTokenizer(corpus_samples=math_corpus, min_frequency=5)
code_tok = CodeTokenizer(corpus_samples=code_corpus)

# Save learned patterns
import pickle
with open('learned_patterns.pkl', 'wb') as f:
    pickle.dump({
        'math_commands': math_tok.LATEX_COMMANDS,
        'code_keywords': code_tok.KEYWORDS
    }, f)
```

### Incremental Learning

```python
tokenizer = MathTokenizer()

# Process documents one by one
for doc in documents:
    tokens = tokenizer.tokenize(doc, learn=True)  # Learns as it goes
    process(tokens)

print(f"Learned {len(tokenizer.LATEX_COMMANDS)} commands")
```

### Disable Learning (Inference Mode)

```python
# After training, freeze the tokenizer
tokenizer = MathTokenizer(corpus_samples=training_data)

# Don't learn from test data
test_tokens = tokenizer.tokenize(test_doc, learn=False)
```

## Pattern Recognition Algorithms

### Keyword Detection Heuristics

1. **Frequency**: Appears multiple times (≥3)
2. **Length**: 2-10 characters
3. **Casing**: Lowercase or special (True/False)
4. **Position**: At statement boundaries
5. **Context**: Not used as identifier

### LaTeX Command Detection

1. **Pattern**: Starts with backslash `\`
2. **Structure**: Alphabetic characters
3. **Usage**: Followed by braces `{` or arguments
4. **Frequency**: Appears in multiple contexts

### Built-in Function Detection

1. **Called with parentheses**: `func(...)`
2. **No definition found**: Not after `def`/`function`
3. **Common usage**: Used across files
4. **Standard library patterns**: Matches stdlib naming

## Comparison: Static vs Dynamic

| Feature | Static (Old) | Dynamic (New) |
|---------|-------------|---------------|
| Keywords | Hardcoded 200+ | Learned from corpus |
| Language Support | Python/JS/C only | Any language |
| Customization | Edit source code | Provide corpus |
| Maintenance | Manual updates | Automatic |
| Accuracy | Generic | Project-specific |
| New Syntax | Requires code change | Auto-learns |

## Migration Guide

### Before (Static)
```python
# Had to use predefined keywords
tokenizer = CodeTokenizer(language="python")
# Only worked with Python/JS/C keywords
```

### After (Dynamic)
```python
# Learn from your actual code
tokenizer = CodeTokenizer(corpus_samples=my_codebase)
# Works with ANY language, even custom DSLs!
```

## Performance Considerations

- **Initial learning**: O(n) where n = corpus size
- **Incremental learning**: O(1) per new pattern
- **Pattern matching**: O(m) where m = learned patterns
- **Memory**: Scales with unique patterns (~1KB per 100 patterns)

## Best Practices

1. **Use corpus training** for production systems
2. **Enable learning** during development
3. **Disable learning** for consistent inference
4. **Save learned patterns** to avoid retraining
5. **Use min_frequency** to filter noise

---

No more hardcoded lists! The tokenizers now learn from your actual data. 🎉
