"""
Unit tests for HMTT (Hybrid Math-Text Tokenizer).

Tests all components: partitioning, tokenization, training, and evaluation.
"""

import pytest
import sys
from pathlib import Path

# Add HMTT to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HMTT.preprocessing.partitioner import TextPartitioner, partition_text
from HMTT.preprocessing.math_tokenizer import MathTokenizer
from HMTT.preprocessing.code_tokenizer import CodeTokenizer
from HMTT.preprocessing.nl_tokenizer import NLTokenizer
from HMTT.training.corpus_builder import CorpusBuilder
from HMTT.evaluation.tfs_metric import TFSEvaluator, compute_tfs


class TestPartitioner:
    """Tests for text partitioning."""
    
    def test_partition_math_inline(self):
        """Test inline math detection."""
        text = "The equation $x^2 + y^2 = r^2$ is a circle."
        partitioner = TextPartitioner()
        regions = partitioner.partition(text)
        
        assert len(regions) == 3
        assert regions[0].type == "nl"
        assert regions[1].type == "math"
        assert regions[2].type == "nl"
        assert "x^2 + y^2 = r^2" in regions[1].text
    
    def test_partition_math_display(self):
        """Test display math detection."""
        text = "Consider the equation: $$\\int_0^1 x^2 dx = \\frac{1}{3}$$"
        partitioner = TextPartitioner()
        regions = partitioner.partition(text)
        
        # Should have NL and MATH regions
        math_regions = [r for r in regions if r.type == "math"]
        assert len(math_regions) >= 1
    
    def test_partition_code_block(self):
        """Test code block detection."""
        text = """Here is some code:
```python
def hello():
    print("Hello, world!")
```
That's it."""
        partitioner = TextPartitioner()
        regions = partitioner.partition(text)
        
        code_regions = [r for r in regions if r.type == "code"]
        assert len(code_regions) >= 1
        assert "def hello" in code_regions[0].text
    
    def test_partition_mixed(self):
        """Test mixed content."""
        text = "Calculate $\\sum_{i=1}^n i$ using ```python sum(range(n+1))```"
        partitioner = TextPartitioner()
        regions = partitioner.partition(text)
        
        math_regions = [r for r in regions if r.type == "math"]
        code_regions = [r for r in regions if r.type == "code"]
        
        assert len(math_regions) >= 1
        assert len(code_regions) >= 1


class TestMathTokenizer:
    """Tests for math tokenization."""
    
    def test_tokenize_latex_command(self):
        """Test LaTeX command tokenization."""
        tokenizer = MathTokenizer()
        tokens = tokenizer.tokenize(r"\frac{1}{2}")
        
        assert r"\frac" in tokens
        assert "{1}" in tokens
        assert "{2}" in tokens
    
    def test_tokenize_variable_subscript(self):
        """Test variable with subscript."""
        tokenizer = MathTokenizer()
        tokens = tokenizer.tokenize("x_i")
        
        # Should keep x_i together
        assert any("x_i" in t or ("x" in tokens and "_" in tokens and "i" in tokens) for t in tokens)
    
    def test_tokenize_number(self):
        """Test number tokenization."""
        tokenizer = MathTokenizer()
        tokens = tokenizer.tokenize("3.14159")
        
        # Number should be atomic
        assert "3.14159" in tokens
    
    def test_tokenize_operators(self):
        """Test operator tokenization."""
        tokenizer = MathTokenizer()
        tokens = tokenizer.tokenize("a + b - c")
        
        assert "+" in tokens
        assert "-" in tokens
    
    def test_is_atomic(self):
        """Test atomic token detection."""
        tokenizer = MathTokenizer()
        
        assert tokenizer.is_atomic(r"\alpha")
        assert tokenizer.is_atomic("123")
        assert tokenizer.is_atomic("x_i")


class TestCodeTokenizer:
    """Tests for code tokenization."""
    
    def test_tokenize_python(self):
        """Test Python code tokenization."""
        tokenizer = CodeTokenizer(language="python")
        tokens = tokenizer.tokenize("def hello():")
        
        assert "def" in tokens
        assert "hello" in tokens
        assert "(" in tokens
        assert ")" in tokens
        assert ":" in tokens
    
    def test_tokenize_number(self):
        """Test number tokenization in code."""
        tokenizer = CodeTokenizer()
        tokens = tokenizer.tokenize("x = 42")
        
        assert "42" in tokens
    
    def test_tokenize_string(self):
        """Test string literal tokenization."""
        tokenizer = CodeTokenizer()
        tokens = tokenizer.tokenize('s = "hello"')
        
        string_tokens = [t for t in tokens if "hello" in t]
        assert len(string_tokens) == 1
    
    def test_is_keyword(self):
        """Test keyword detection."""
        tokenizer = CodeTokenizer()
        
        assert tokenizer.is_keyword("def")
        assert tokenizer.is_keyword("class")
        assert not tokenizer.is_keyword("hello")
    
    def test_is_literal(self):
        """Test literal detection."""
        tokenizer = CodeTokenizer()
        
        assert tokenizer.is_literal("123")
        assert tokenizer.is_literal('"hello"')


class TestNLTokenizer:
    """Tests for natural language tokenization."""
    
    def test_tokenize_simple(self):
        """Test simple tokenization."""
        tokenizer = NLTokenizer(use_unicode=False)
        tokens = tokenizer.tokenize("Hello, world!")
        
        assert "Hello" in tokens or "hello" in [t.lower() for t in tokens]
        assert "world" in tokens or "world" in [t.lower() for t in tokens]
    
    def test_tokenize_contractions(self):
        """Test contraction handling."""
        tokenizer = NLTokenizer(use_unicode=False)
        tokens = tokenizer.tokenize("don't")
        
        # Should handle contractions
        assert len(tokens) >= 1
    
    def test_is_number(self):
        """Test number detection."""
        tokenizer = NLTokenizer()
        
        assert tokenizer.is_number("123")
        assert tokenizer.is_number("3.14")
        assert not tokenizer.is_number("hello")
    
    def test_is_punctuation(self):
        """Test punctuation detection."""
        tokenizer = NLTokenizer()
        
        assert tokenizer.is_punctuation(".")
        assert tokenizer.is_punctuation("!")
        assert not tokenizer.is_punctuation("hello")


class TestCorpusBuilder:
    """Tests for corpus building."""
    
    def test_process_document(self):
        """Test document processing."""
        builder = CorpusBuilder()
        text = "Calculate $x^2$ using ```python x**2```"
        tokens = builder.process_document(text)
        
        assert len(tokens) > 0
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)
    
    def test_process_documents(self):
        """Test batch document processing."""
        builder = CorpusBuilder()
        documents = [
            "First document with $math$",
            "Second document with ```code```"
        ]
        results = builder.process_documents(documents)
        
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)


class TestTFSMetric:
    """Tests for TFS metric."""
    
    def test_compute_tfs_perfect(self):
        """Test TFS with perfect tokenization."""
        text = "hello world"
        tokens = ["hello", " ", "world"]
        
        metrics = compute_tfs(text, tokens)
        
        assert metrics.tfs_score >= 0.0
        assert metrics.tfs_score <= 1.0
    
    def test_compute_tfs_split_number(self):
        """Test TFS with split number."""
        text = "The number 123 is here"
        tokens = ["The", " ", "number", " ", "1", "2", "3", " ", "is", " ", "here"]
        
        metrics = compute_tfs(text, tokens)
        
        # Should have lower score due to split number
        assert metrics.atomic_splits > 0
    
    def test_compute_tfs_latex(self):
        """Test TFS with LaTeX."""
        text = r"The command \alpha is Greek"
        tokens = [r"The", " ", "command", " ", r"\alpha", " ", "is", " ", "Greek"]
        
        metrics = compute_tfs(text, tokens)
        
        # Should preserve LaTeX command
        assert metrics.tfs_score > 0.5
    
    def test_evaluator(self):
        """Test TFS evaluator."""
        evaluator = TFSEvaluator()
        
        text = "Test text with $x^2$"
        tokens = ["Test", " ", "text", " ", "with", " ", "x", "^", "2"]
        
        metrics = evaluator.compute_tfs(text, tokens)
        
        assert isinstance(metrics.tfs_score, float)
        assert metrics.total_tokens == len(tokens)


def test_integration():
    """Integration test: full pipeline."""
    # Create sample text
    text = "Calculate the sum $\\sum_{i=1}^n i = \\frac{n(n+1)}{2}$ using Python: ```python sum(range(n+1))```"
    
    # Partition
    partitioner = TextPartitioner()
    regions = partitioner.partition(text)
    assert len(regions) > 0
    
    # Build corpus
    builder = CorpusBuilder()
    tokens = builder.process_document(text)
    assert len(tokens) > 0
    
    # Evaluate
    metrics = compute_tfs(text, tokens)
    assert metrics.tfs_score >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
