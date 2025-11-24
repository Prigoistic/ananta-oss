"""
Corpus Builder for HMTT.

Processes documents through partitioning and domain-specific tokenization
to create a training corpus for BPE vocabulary learning.
"""

from typing import List, Dict, Iterator, Optional
from pathlib import Path
import json

from ..preprocessing.partitioner import TextPartitioner
from ..preprocessing.math_tokenizer import MathTokenizer
from ..preprocessing.code_tokenizer import CodeTokenizer
from ..preprocessing.nl_tokenizer import NLTokenizer


class CorpusBuilder:
    """
    Builds a training corpus from documents.
    
    Process:
    1. Partition each document into NL/MATH/CODE regions
    2. Apply domain-specific tokenizer to each region
    3. Concatenate pre-tokens into a unified sequence
    4. Write sequences to corpus file (one per line)
    """
    
    def __init__(
        self,
        code_language: str = "python",
        use_unicode_nl: bool = True
    ):
        """
        Initialize the corpus builder.
        
        Args:
            code_language: Programming language for code tokenizer
            use_unicode_nl: Use unicode-aware NL tokenization
        """
        self.partitioner = TextPartitioner()
        self.math_tokenizer = MathTokenizer()
        self.code_tokenizer = CodeTokenizer(language=code_language)
        self.nl_tokenizer = NLTokenizer(use_unicode=use_unicode_nl)
    
    def process_document(self, text: str) -> List[str]:
        """
        Process a single document into pre-tokens.
        
        Args:
            text: Document text
            
        Returns:
            List of pre-token strings
        """
        # Partition text into regions
        regions = self.partitioner.partition_to_dict(text)
        
        # Tokenize each region
        all_tokens = []
        
        for region in regions:
            region_type = region["type"]
            region_text = region["text"]
            
            if region_type == "math":
                tokens = self.math_tokenizer.tokenize(region_text)
            elif region_type == "code":
                tokens = self.code_tokenizer.tokenize(region_text)
            else:  # nl
                tokens = self.nl_tokenizer.tokenize(region_text)
            
            all_tokens.extend(tokens)
        
        return all_tokens
    
    def process_documents(
        self,
        documents: List[str]
    ) -> List[List[str]]:
        """
        Process multiple documents.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of token sequences (one per document)
        """
        return [self.process_document(doc) for doc in documents]
    
    def build_corpus(
        self,
        documents: List[str],
        output_path: str,
        verbose: bool = False
    ) -> int:
        """
        Build a corpus file from documents.
        
        Args:
            documents: List of document texts
            output_path: Path to output corpus file
            verbose: Print progress
            
        Returns:
            Number of documents processed
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, doc in enumerate(documents):
                if verbose and i % 100 == 0:
                    print(f"Processed {i}/{len(documents)} documents")
                
                tokens = self.process_document(doc)
                # Write tokens as space-separated line
                f.write(' '.join(tokens) + '\n')
        
        if verbose:
            print(f"Corpus built: {len(documents)} documents -> {output_path}")
        
        return len(documents)
    
    def build_corpus_from_files(
        self,
        input_files: List[str],
        output_path: str,
        verbose: bool = False
    ) -> int:
        """
        Build a corpus from text files.
        
        Args:
            input_files: List of input file paths
            output_path: Path to output corpus file
            verbose: Print progress
            
        Returns:
            Number of files processed
        """
        documents = []
        
        for file_path in input_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    documents.append(f.read())
            except Exception as e:
                if verbose:
                    print(f"Error reading {file_path}: {e}")
        
        return self.build_corpus(documents, output_path, verbose)
    
    def build_corpus_from_jsonl(
        self,
        jsonl_path: str,
        output_path: str,
        text_field: str = "text",
        verbose: bool = False
    ) -> int:
        """
        Build a corpus from JSONL file.
        
        Args:
            jsonl_path: Path to JSONL file
            output_path: Path to output corpus file
            text_field: Field name containing text
            verbose: Print progress
            
        Returns:
            Number of documents processed
        """
        documents = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    if text_field in data:
                        documents.append(data[text_field])
                except Exception as e:
                    if verbose:
                        print(f"Error parsing line {i}: {e}")
        
        return self.build_corpus(documents, output_path, verbose)
    
    def stream_corpus(
        self,
        documents: Iterator[str],
        output_path: str,
        verbose: bool = False,
        batch_size: int = 1000
    ) -> int:
        """
        Build corpus from a document stream (memory efficient).
        
        Args:
            documents: Iterator of document texts
            output_path: Path to output corpus file
            verbose: Print progress
            batch_size: Number of documents to process before writing
            
        Returns:
            Number of documents processed
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        count = 0
        batch = []
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in documents:
                tokens = self.process_document(doc)
                batch.append(' '.join(tokens))
                count += 1
                
                if len(batch) >= batch_size:
                    f.write('\n'.join(batch) + '\n')
                    batch = []
                    
                    if verbose and count % 1000 == 0:
                        print(f"Processed {count} documents")
            
            # Write remaining batch
            if batch:
                f.write('\n'.join(batch) + '\n')
        
        if verbose:
            print(f"Corpus built: {count} documents -> {output_path}")
        
        return count
    
    def get_vocabulary_preview(
        self,
        documents: List[str],
        top_k: int = 100
    ) -> Dict[str, int]:
        """
        Get a preview of token frequencies in documents.
        
        Args:
            documents: List of document texts
            top_k: Number of top tokens to return
            
        Returns:
            Dictionary of token -> frequency
        """
        from collections import Counter
        
        token_counts = Counter()
        
        for doc in documents:
            tokens = self.process_document(doc)
            token_counts.update(tokens)
        
        return dict(token_counts.most_common(top_k))
