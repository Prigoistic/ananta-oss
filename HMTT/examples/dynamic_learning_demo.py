"""
Example: Dynamic Learning Tokenizers

Demonstrates how HMTT tokenizers learn patterns from corpus
without any hardcoded keywords or commands.
"""

from HMTT.preprocessing import MathTokenizer, CodeTokenizer
from HMTT.preprocessing.dynamic_analyzer import (
    DynamicMathAnalyzer,
    DynamicCodeAnalyzer,
    SemanticTokenClassifier
)


def example_math_dynamic_learning():
    """Example: Math tokenizer learns LaTeX commands from corpus."""
    print("=== Math Tokenizer Dynamic Learning ===\n")
    
    # Corpus with various LaTeX commands
    math_corpus = [
        r"\frac{x^2 + y^2}{2}",
        r"\int_0^\infty e^{-x} dx",
        r"\sum_{i=1}^n \alpha_i x_i",
        r"\lim_{x \to \infty} f(x)",
        r"\sqrt{\beta^2 - 4ac}",
        # Custom commands
        r"\mycustomop{a} + \specialfunc{b}",
        r"\mycustomop{c} - \specialfunc{d}",
    ]
    
    # Create tokenizer with corpus
    print("Training tokenizer on corpus...")
    tokenizer = MathTokenizer(corpus_samples=math_corpus, min_frequency=2)
    
    print(f"Learned {len(tokenizer.LATEX_COMMANDS)} LaTeX commands")
    print(f"Commands: {sorted(tokenizer.LATEX_COMMANDS)}\n")
    
    # Test on new expression
    test_expr = r"\mycustomop{x} + \int_0^1 \alpha dx"
    tokens = tokenizer.tokenize(test_expr, learn=False)
    print(f"Test expression: {test_expr}")
    print(f"Tokens: {tokens}\n")
    
    # Learn new command on-the-fly
    print("Learning new command on-the-fly...")
    new_expr = r"\brandnewcmd{z}"
    tokens = tokenizer.tokenize(new_expr, learn=True)
    print(f"New expression: {new_expr}")
    print(f"Tokens: {tokens}")
    print(f"Now knows {len(tokenizer.LATEX_COMMANDS)} commands\n")


def example_code_dynamic_learning():
    """Example: Code tokenizer learns keywords from corpus."""
    print("=== Code Tokenizer Dynamic Learning ===\n")
    
    # Corpus with Python-like code
    code_corpus = [
        "def hello(): return 42",
        "def greet(name): print(name)",
        "class MyClass: pass",
        "for i in range(10): print(i)",
        "if x > 0: return True",
        "while running: process()",
        "import os",
        "from math import sqrt",
    ]
    
    # Create tokenizer with corpus
    print("Training tokenizer on Python corpus...")
    tokenizer = CodeTokenizer(language="python", corpus_samples=code_corpus)
    
    print(f"Learned {len(tokenizer.KEYWORDS)} keywords")
    print(f"Keywords: {sorted(tokenizer.KEYWORDS)}")
    print(f"Builtins: {sorted(tokenizer.BUILTINS)}\n")
    
    # Test on new code
    test_code = "def factorial(n): return 1 if n == 0 else n * factorial(n-1)"
    tokens = tokenizer.tokenize(test_code, learn=False)
    print(f"Test code: {test_code}")
    print(f"Tokens: {tokens[:15]}...\n")


def example_custom_language():
    """Example: Learn custom DSL without any predefined keywords."""
    print("=== Custom DSL Learning ===\n")
    
    # Fictional DSL corpus
    dsl_corpus = [
        "fnc greet(name): show(name)",
        "fnc add(x, y): ret x + y",
        "loop i in 1..10: show(i)",
        "cond x > 0: ret true",
        "cond x < 0: ret false",
    ]
    
    print("Training tokenizer on custom DSL...")
    tokenizer = CodeTokenizer(language="custom", corpus_samples=dsl_corpus)
    
    print(f"Learned DSL keywords: {sorted(tokenizer.KEYWORDS)}")
    print(f"Learned builtins: {sorted(tokenizer.BUILTINS)}\n")
    
    # Test on new DSL code
    test_dsl = "fnc multiply(a, b): ret a * b"
    tokens = tokenizer.tokenize(test_dsl, learn=False)
    print(f"Test DSL code: {test_dsl}")
    print(f"Tokens: {tokens}\n")


def example_analyzer_direct():
    """Example: Use analyzers directly for pattern discovery."""
    print("=== Direct Analyzer Usage ===\n")
    
    # Math analyzer
    print("Math Pattern Discovery:")
    math_analyzer = DynamicMathAnalyzer()
    patterns = math_analyzer.analyze_corpus([
        r"\alpha + \beta = \gamma",
        r"\myop{x} + \myop{y}",
        r"\sin(\theta) + \cos(\theta)",
    ], min_frequency=1)
    
    print(f"Discovered commands: {patterns['commands']}")
    print(f"Discovered symbols: {patterns['symbols']}\n")
    
    # Code analyzer
    print("Code Pattern Discovery:")
    code_analyzer = DynamicCodeAnalyzer()
    patterns = code_analyzer.analyze_corpus([
        "procedure greet(): display('Hello')",
        "procedure main(): greet()",
        "loop i from 1 to 10: display(i)",
    ])
    
    print(f"Discovered keywords: {patterns['keywords']}")
    print(f"Control flow: {patterns['control_flow']}")
    print(f"Definitions: {patterns['definitions']}\n")


def example_semantic_classifier():
    """Example: Semantic token classification."""
    print("=== Semantic Token Classifier ===\n")
    
    classifier = SemanticTokenClassifier()
    
    # Teach classifier
    tokens = ['def', 'hello', '(', ')', ':', 'return', '42']
    contexts = ['keyword', 'identifier', 'operator', 'operator', 
                'operator', 'keyword', 'literal']
    
    classifier.learn_from_context(tokens, contexts)
    
    # Test classification
    test_tokens = ['def', 'hello', '42', 'unknown_token']
    print("Token classifications:")
    for token in test_tokens:
        classification = classifier.classify_token(token)
        print(f"  {token:15} -> {classification}")
    print()


def example_incremental_learning():
    """Example: Incremental learning as documents are processed."""
    print("=== Incremental Learning ===\n")
    
    tokenizer = MathTokenizer()  # Start empty
    print(f"Initial commands: {len(tokenizer.LATEX_COMMANDS)}")
    
    # Process documents one by one
    documents = [
        r"\alpha + \beta",
        r"\gamma - \delta",
        r"\int_0^1 f(x) dx",
        r"\sum_{i=1}^n x_i",
    ]
    
    for i, doc in enumerate(documents, 1):
        tokenizer.tokenize(doc, learn=True)
        print(f"After doc {i}: {len(tokenizer.LATEX_COMMANDS)} commands")
    
    print(f"\nFinal learned commands: {sorted(tokenizer.LATEX_COMMANDS)}\n")


def main():
    """Run all examples."""
    print("HMTT Dynamic Learning Examples")
    print("=" * 60)
    print()
    
    example_math_dynamic_learning()
    print("-" * 60)
    print()
    
    example_code_dynamic_learning()
    print("-" * 60)
    print()
    
    example_custom_language()
    print("-" * 60)
    print()
    
    example_analyzer_direct()
    print("-" * 60)
    print()
    
    example_semantic_classifier()
    print("-" * 60)
    print()
    
    example_incremental_learning()
    print("-" * 60)
    print()
    
    print("✅ All examples completed!")
    print("\nKey Takeaway: No hardcoded keywords - everything is learned!")


if __name__ == "__main__":
    main()
