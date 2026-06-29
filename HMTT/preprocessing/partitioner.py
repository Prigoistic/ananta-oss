"""
Text Partitioner for HMTT.

Segments input text into natural language (NL), mathematical (MATH), and code (CODE) regions.
"""

import re
from typing import List, Dict, Literal
from dataclasses import dataclass


@dataclass
class Region:
    """Represents a segmented region of text."""
    type: Literal["nl", "math", "code"]
    text: str
    start: int
    end: int


class TextPartitioner:
    """
    Partitions text into NL, MATH, and CODE regions.
    
    Math delimiters:
    - Inline: $...$
    - Display: $$...$$, \\[...\\], \\begin{equation}...\\end{equation}
    
    Code delimiters:
    - Inline: `...`
    - Block: ```...```
    """
    
    # Math patterns
    DISPLAY_MATH_PATTERNS = [
        (r'\$\$(.+?)\$\$', 'display'),  # $$...$$
        (r'\\\[(.+?)\\\]', 'display'),  # \[...\]
        (r'\\begin\{equation\}(.+?)\\end\{equation\}', 'display'),
        (r'\\begin\{align\}(.+?)\\end\{align\}', 'display'),
        (r'\\begin\{eqnarray\}(.+?)\\end\{eqnarray\}', 'display'),
        (r'\\begin\{gather\}(.+?)\\end\{gather\}', 'display'),
        (r'\\begin\{multline\}(.+?)\\end\{multline\}', 'display'),
    ]
    
    INLINE_MATH_PATTERN = r'\$([^\$]+?)\$'  # $...$
    
    # Code patterns
    CODE_BLOCK_PATTERN = r'```[\w]*\n(.+?)\n```'  # ```...```
    INLINE_CODE_PATTERN = r'`([^`]+?)`'  # `...`
    
    def __init__(self):
        """Initialize the text partitioner."""
        pass
    
    def partition(self, text: str) -> List[Region]:
        """
        Partition text into regions.
        
        Args:
            text: Input text to partition
            
        Returns:
            List of Region objects ordered by appearance
        """
        regions = []
        
        # Find all math regions (display first, then inline)
        for pattern, _ in self.DISPLAY_MATH_PATTERNS:
            for match in re.finditer(pattern, text, re.DOTALL):
                regions.append(Region(
                    type="math",
                    text=match.group(1),
                    start=match.start(),
                    end=match.end()
                ))
        
        for match in re.finditer(self.INLINE_MATH_PATTERN, text):
            regions.append(Region(
                type="math",
                text=match.group(1),
                start=match.start(),
                end=match.end()
            ))
        
        # Find all code regions (block first, then inline)
        for match in re.finditer(self.CODE_BLOCK_PATTERN, text, re.DOTALL):
            regions.append(Region(
                type="code",
                text=match.group(1),
                start=match.start(),
                end=match.end()
            ))
        
        for match in re.finditer(self.INLINE_CODE_PATTERN, text):
            regions.append(Region(
                type="code",
                text=match.group(1),
                start=match.start(),
                end=match.end()
            ))
        
        # Sort regions by start position
        regions.sort(key=lambda r: r.start)
        
        # Fill in NL regions between math and code
        filled_regions = []
        last_end = 0
        
        for region in regions:
            # Check for overlap (prefer longer/earlier regions)
            if region.start < last_end:
                continue
                
            # Add NL region before this one if there's a gap
            if region.start > last_end:
                nl_text = text[last_end:region.start]
                if nl_text.strip():  # Only add non-empty NL regions
                    filled_regions.append(Region(
                        type="nl",
                        text=nl_text,
                        start=last_end,
                        end=region.start
                    ))
            
            filled_regions.append(region)
            last_end = region.end
        
        # Add final NL region if text remains
        if last_end < len(text):
            nl_text = text[last_end:]
            if nl_text.strip():
                filled_regions.append(Region(
                    type="nl",
                    text=nl_text,
                    start=last_end,
                    end=len(text)
                ))
        
        # If no regions found, treat entire text as NL
        if not filled_regions:
            filled_regions.append(Region(
                type="nl",
                text=text,
                start=0,
                end=len(text)
            ))
        
        return filled_regions
    
    def partition_to_dict(self, text: str) -> List[Dict[str, str]]:
        """
        Partition text and return as list of dictionaries.
        
        Args:
            text: Input text to partition
            
        Returns:
            List of dicts with 'type' and 'text' keys
        """
        regions = self.partition(text)
        return [{"type": region.type, "text": region.text} for region in regions]


def partition_text(text: str) -> List[Dict[str, str]]:
    """
    Convenience function to partition text.
    
    Args:
        text: Input text to partition
        
    Returns:
        List of dicts with 'type' and 'text' keys
    """
    partitioner = TextPartitioner()
    return partitioner.partition_to_dict(text)
