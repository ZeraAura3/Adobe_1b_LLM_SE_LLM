#!/usr/bin/env python3
"""
Round 1B: PDF Processor with Section-to-Text Mapping
Builds upon Round 1A heading extraction for persona-driven document intelligence
"""

try:
    import fitz  # PyMuPDF - fastest PDF library
except ImportError:
    print("PyMuPDF not available, falling back to PyPDF2")
    fitz = None

import json
import os
import re
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        # Adjusted weights based on sample analysis
        self.weights = {
            'font_size': 0.40,
            'bold': 0.30,
            'position': 0.15,
            'whitespace': 0.10,
            'pattern': 0.05
        }
        
        # More precise heading patterns
        self.heading_patterns = [
            r'^\d+\.\s+[A-Z]',           # "1. Introduction"
            r'^\d+\.\d+\s+[A-Z]',        # "1.1 Background"
            r'^\d+\.\d+\.\d+\s+[A-Z]',   # "1.1.1 Details"
            r'^[A-Z][A-Z\s]{3,}:?\s*$',  # ALL CAPS headings
            r'^\d+\s+[A-Z][a-z]',        # "1 Introduction"
            r'^[A-Z]\.\s+[A-Z]',         # "A. Section"
            r'^\d+\.\s*$',               # Just numbers "1."
            r'^(Chapter|Section|Part|Appendix)\s+\d+',
            r'^[IVX]+\.\s+',             # Roman numerals
        ]
        
        # Words that indicate headings
        self.heading_indicators = [
            'introduction', 'background', 'summary', 'conclusion', 'methodology',
            'results', 'discussion', 'references', 'abstract', 'overview',
            'acknowledgements', 'table of contents', 'revision history',
            'appendix', 'bibliography', 'index'
        ]
        
        # Words/patterns that indicate NOT headings
        self.non_heading_indicators = [
            'page', 'figure', 'table', 'equation', 'note:', 'example:',
            'copyright', '©', 'all rights reserved', 'www.', 'http',
            'email', '@', '.com', '.org', '.net'
        ]

    def extract_text_blocks_fast(self, pdf_path):
        """Fast text extraction with essential metadata"""
        try:
            doc = fitz.open(pdf_path)
            text_blocks = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Use get_text("dict") for detailed formatting info
                blocks = page.get_text("dict")
                
                for block in blocks["blocks"]:
                    if "lines" not in block:
                        continue
                        
                    for line in block["lines"]:
                        line_text = ""
                        max_font_size = 0
                        is_bold = False
                        font_name = ""
                        
                        # Combine spans in the same line
                        for span in line["spans"]:
                            line_text += span["text"]
                            if span["size"] > max_font_size:
                                max_font_size = span["size"]
                                font_name = span["font"]
                                is_bold = span["flags"] & 16  # Bold flag
                        
                        line_text = line_text.strip()
                        if len(line_text) > 2:  # Skip very short text
                            text_blocks.append({
                                'text': line_text,
                                'font_size': max_font_size,
                                'is_bold': is_bold,
                                'font_name': font_name,
                                'page': page_num + 1,
                                'bbox': line["bbox"],
                                'x0': line["bbox"][0],
                                'y0': line["bbox"][1],
                                'x1': line["bbox"][2],
                                'y1': line["bbox"][3]
                            })
            
            doc.close()
            return text_blocks
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return []

    def analyze_font_distribution(self, text_blocks):
        """Quick font analysis for adaptive thresholds"""
        font_sizes = [b['font_size'] for b in text_blocks if len(b['text']) > 3]
        
        if not font_sizes:
            return {'percentiles': [10, 12, 14, 16]}
            
        return {
            'mean': np.mean(font_sizes),
            'std': np.std(font_sizes),
            'percentiles': np.percentile(font_sizes, [50, 75, 90, 95]),
            'max': np.max(font_sizes)
        }

    def calculate_whitespace_score(self, text_blocks, idx):
        """Calculate whitespace above current block"""
        if idx == 0:
            return 0.5  # First block gets medium score
            
        current = text_blocks[idx]
        prev = text_blocks[idx - 1]
        
        # Different page = high whitespace
        if current['page'] != prev['page']:
            return 1.0
            
        # Calculate vertical gap
        gap = current['y0'] - prev['y1']
        
        # Normalize gap (typical line height ~15-20)
        if gap > 30:
            return 1.0
        elif gap > 15:
            return 0.7
        elif gap > 5:
            return 0.3
        else:
            return 0.0

    def is_likely_heading(self, text):
        """Determine if text is likely a heading based on content"""
        text_lower = text.lower().strip()
        
        # Check for non-heading indicators first
        for indicator in self.non_heading_indicators:
            if indicator in text_lower:
                return False
        
        # Form-specific exclusions - only for obvious form elements
        form_exclusions = [
            r'^\d+\.$',  # Just numbers like "1.", "2."
            r'^\d+\.\s*$',  # Numbers with spaces
            r'^s\.no$',  # Serial number
            r'^name$',   # Single word field labels
            r'^age$',
            r'^date$',
            r'^relationship$',
            r'^designation$',
            r'^amount$',
            r'^[a-z]+:?\s*$',  # Single lowercase words
        ]
        
        for pattern in form_exclusions:
            if re.match(pattern, text_lower):
                return False
        
        # Skip very short standalone text that's likely form fields
        if len(text) < 5 and len(text.split()) <= 1:  # Only single very short words
            return False
        
        # Check length and structure - STRICTER RULES
        if len(text) > 100:  # Much shorter limit for headings
            return False
        
        # Check for specific heading patterns FIRST (most reliable)
        for pattern in self.heading_patterns:
            if re.match(pattern, text):
                return True
        
        # Check for heading indicator words (but be more strict)
        heading_word_found = False
        for indicator in self.heading_indicators:
            if indicator in text_lower:
                heading_word_found = True
                break
        
        # If it has heading words, it must also be relatively short and not sentence-like
        if heading_word_found:
            # Must be short and not end with a period (unless it's a numbered heading)
            if len(text) <= 50 and (not text.endswith('.') or re.match(r'^\d+\.', text)):
                return True
        
        # Additional checks for ALL CAPS (but not if it's a long sentence)
        if text.isupper() and len(text) <= 30 and len(text.split()) <= 5:
            return True
        
        # If none of the above criteria are met, it's probably not a heading
        return False

    def calculate_heading_scores(self, text_blocks):
        """Calculate heading scores for all text blocks"""
        font_stats = self.analyze_font_distribution(text_blocks)
        
        for idx, block in enumerate(text_blocks):
            block['is_heading_candidate'] = self.is_likely_heading(block['text'])
            
            # Font size score (higher = more likely to be heading)
            font_score = min(1.0, block['font_size'] / font_stats['percentiles'][2])
            
            # Bold score
            bold_score = 1.0 if block['is_bold'] else 0.0
            
            # Position score (left alignment)
            position_score = 1.0 - min(1.0, block['x0'] / 100)  # Normalize to page width
            
            # Whitespace score
            whitespace_score = self.calculate_whitespace_score(text_blocks, idx)
            
            # Pattern score
            pattern_score = 1.0 if self.is_likely_heading(block['text']) else 0.0
            
            # Combined score
            block['heading_score'] = (
                self.weights['font_size'] * font_score +
                self.weights['bold'] * bold_score +
                self.weights['position'] * position_score +
                self.weights['whitespace'] * whitespace_score +
                self.weights['pattern'] * pattern_score
            )

    def extract_headings(self, text_blocks, threshold=0.4):
        """Extract headings based on scores"""
        headings = []
        
        for block in text_blocks:
            if block['heading_score'] >= threshold and len(block['text'].strip()) > 2:
                headings.append({
                    'text': block['text'].strip(),
                    'page': block['page'],
                    'level': self.determine_heading_level(block),
                    'score': block['heading_score'],
                    'bbox': block['bbox']
                })
        
        return headings

    def determine_heading_level(self, block):
        """Determine heading level based on patterns and formatting"""
        text = block['text'].strip()
        
        # Level 1: Major sections
        if re.match(r'^\d+\.\s+[A-Z]', text) or re.match(r'^[A-Z][A-Z\s]{3,}:?\s*$', text):
            return 1
        
        # Level 2: Subsections
        if re.match(r'^\d+\.\d+\s+[A-Z]', text):
            return 2
        
        # Level 3: Sub-subsections
        if re.match(r'^\d+\.\d+\.\d+\s+[A-Z]', text):
            return 3
        
        # Default based on font size relative to document
        if block['font_size'] >= 16:
            return 1
        elif block['font_size'] >= 14:
            return 2
        else:
            return 3

    def extract_text_with_structure(self, pdf_path):
        """
        Extract text with section mapping for Round 1B
        Returns: {sections: [{title, text, page, level}]}
        """
        try:
            text_blocks = self.extract_text_blocks_fast(pdf_path)
            if not text_blocks:
                return {'sections': []}
            
            # Calculate heading scores
            self.calculate_heading_scores(text_blocks)
            
            # Extract headings
            headings = self.extract_headings(text_blocks)
            
            # Map text to sections
            sections = self.map_text_to_sections(text_blocks, headings)
            
            return {
                'document': os.path.basename(pdf_path),
                'sections': sections
            }
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return {'sections': []}

    def map_text_to_sections(self, text_blocks, headings):
        """Map text content to sections based on headings"""
        sections = []
        
        if not headings:
            # No headings found, treat entire document as one section
            all_text = '\n'.join([block['text'] for block in text_blocks])
            sections.append({
                'title': 'Document Content',
                'text': all_text,
                'page': 1,
                'level': 1,
                'start_page': 1,
                'end_page': max([block['page'] for block in text_blocks]) if text_blocks else 1
            })
            return sections
        
        # Create section mapping
        heading_positions = {}
        for i, heading in enumerate(headings):
            # Find the position of this heading in text_blocks
            for j, block in enumerate(text_blocks):
                if (block['text'].strip() == heading['text'] and 
                    block['page'] == heading['page']):
                    heading_positions[i] = j
                    break
        
        # Extract text for each section
        for i, heading in enumerate(headings):
            section_text = []
            start_pos = heading_positions.get(i, 0)
            
            # Find end position (next heading or end of document)
            end_pos = len(text_blocks)
            if i + 1 < len(headings):
                end_pos = heading_positions.get(i + 1, len(text_blocks))
            
            # Collect text between this heading and the next
            current_page = heading['page']
            end_page = current_page
            
            for j in range(start_pos + 1, end_pos):  # Skip the heading itself
                block = text_blocks[j]
                # Skip other headings in between
                if block['heading_score'] < 0.4:  # Not a heading
                    section_text.append(block['text'])
                    end_page = max(end_page, block['page'])
            
            sections.append({
                'title': heading['text'],
                'text': '\n'.join(section_text),
                'page': heading['page'],
                'level': heading['level'],
                'start_page': current_page,
                'end_page': end_page
            })
        
        return sections

    def batch_process_pdfs(self, input_dir):
        """Process all PDFs in directory"""
        all_sections = []
        pdf_files = list(Path(input_dir).glob("*.pdf"))
        
        logger.info(f"Processing {len(pdf_files)} PDF files...")
        
        for pdf_path in pdf_files:
            logger.info(f"Processing: {pdf_path}")
            try:
                result = self.extract_text_with_structure(str(pdf_path))
                
                # Add document info to each section
                for section in result['sections']:
                    section['document'] = result['document']
                    section['pdf_path'] = str(pdf_path)
                
                all_sections.extend(result['sections'])
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                continue
        
        logger.info(f"Extracted {len(all_sections)} sections from {len(pdf_files)} documents")
        return all_sections

    def extract_full_text(self, pdf_path):
        """Extract full text without structure for fallback"""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                full_text += page.get_text()
                full_text += "\n\n"  # Page separator
            
            doc.close()
            return full_text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting full text from {pdf_path}: {e}")
            return ""

def test_pdf_processor():
    """Test PDF processor functionality"""
    print("Testing PDF Processor...")
    
    processor = PDFProcessor()
    
    # Test font analysis
    sample_blocks = [
        {'text': 'Sample text 1', 'font_size': 12, 'is_bold': False},
        {'text': 'Sample text 2', 'font_size': 14, 'is_bold': True},
        {'text': 'Sample text 3', 'font_size': 16, 'is_bold': False}
    ]
    
    font_stats = processor.analyze_font_distribution(sample_blocks)
    print(f"✓ Font analysis: mean={font_stats.get('mean', 0):.1f}")
    
    # Test heading detection
    test_texts = [
        ("1. Introduction", True),  # Should be heading
        ("This is regular paragraph text that should not be a heading.", False),  # Should NOT be heading
        ("METHODOLOGY", True),  # Should be heading (short, all caps)
        ("2.1 Data Collection", True),  # Should be heading
        ("The quick brown fox jumps over the lazy dog in this sentence.", False),  # Should NOT be heading
        ("INTRODUCTION", True),  # Should be heading
        ("Figure 1 shows the results of the experiment conducted.", False),  # Should NOT be heading
        ("3. Results and Discussion", True),  # Should be heading
    ]
    
    correct_count = 0
    total_count = len(test_texts)
    
    for text, expected in test_texts:
        is_heading = processor.is_likely_heading(text)
        if is_heading == expected:
            correct_count += 1
            status = "✓ CORRECT"
        else:
            status = "❌ WRONG"
        
        print(f"{status}: '{text}' -> {'Heading' if is_heading else 'Not heading'} (expected: {'Heading' if expected else 'Not heading'})")
    
    print(f"✓ Heading detection accuracy: {correct_count}/{total_count} ({100*correct_count/total_count:.1f}%)")
    
    # Test section mapping
    sample_headings = [
        {'text': '1. Introduction', 'page': 1, 'level': 1},
        {'text': '2. Methodology', 'page': 2, 'level': 1}
    ]
    
    sample_text_blocks = [
        {'text': '1. Introduction', 'heading_score': 0.8, 'page': 1},
        {'text': 'This is the introduction content.', 'heading_score': 0.1, 'page': 1},
        {'text': '2. Methodology', 'heading_score': 0.8, 'page': 2},
        {'text': 'This is the methodology content.', 'heading_score': 0.1, 'page': 2}
    ]
    
    sections = processor.map_text_to_sections(sample_text_blocks, sample_headings)
    print(f"✓ Created {len(sections)} sections")
    
    print("PDF Processor test completed successfully!")
    return True

def test_with_custom_pdf(pdf_path):
    """Test PDF processor with a custom PDF file"""
    print("=" * 60)
    print(f"Testing PDF Processor with custom PDF: {pdf_path}")
    print("=" * 60)
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found: {pdf_path}")
        return False
    
    processor = PDFProcessor()
    
    try:
        print("\n1. Extracting text blocks...")
        text_blocks = processor.extract_text_blocks_fast(pdf_path)
        print(f"✓ Extracted {len(text_blocks)} text blocks")
        
        if text_blocks:
            print(f"✓ First few text blocks:")
            for i, block in enumerate(text_blocks[:5]):
                print(f"  {i+1}. Page {block['page']}: '{block['text'][:50]}...' (font: {block['font_size']:.1f}, bold: {block['is_bold']})")
        
        print("\n2. Analyzing font distribution...")
        font_stats = processor.analyze_font_distribution(text_blocks)
        print(f"✓ Font statistics:")
        print(f"  Mean font size: {font_stats.get('mean', 0):.1f}")
        print(f"  Font percentiles: {[f'{p:.1f}' for p in font_stats.get('percentiles', [])]}")
        print(f"  Max font size: {font_stats.get('max', 0):.1f}")
        
        print("\n3. Calculating heading scores...")
        processor.calculate_heading_scores(text_blocks)
        high_score_blocks = [b for b in text_blocks if b.get('heading_score', 0) > 0.4]
        print(f"✓ Found {len(high_score_blocks)} potential headings (score > 0.4)")
        
        print("\n4. Extracting headings...")
        headings = processor.extract_headings(text_blocks)
        print(f"✓ Extracted {len(headings)} headings:")
        for i, heading in enumerate(headings[:10]):  # Show first 10 headings
            print(f"  {i+1}. Level {heading['level']}, Page {heading['page']}: '{heading['text']}' (score: {heading['score']:.3f})")
        
        print("\n5. Creating section mapping...")
        sections = processor.map_text_to_sections(text_blocks, headings)
        print(f"✓ Created {len(sections)} sections:")
        for i, section in enumerate(sections[:5]):  # Show first 5 sections
            text_preview = section['text'][:100].replace('\n', ' ') + "..." if len(section['text']) > 100 else section['text']
            print(f"  {i+1}. '{section['title']}' (Page {section['page']}, Level {section['level']})")
            print(f"     Text: {text_preview}")
            print(f"     Length: {len(section['text'])} characters")
        
        print("\n6. Full document structure extraction...")
        result = processor.extract_text_with_structure(pdf_path)
        print(f"✓ Document: {result['document']}")
        print(f"✓ Total sections: {len(result['sections'])}")
        
        print("\n7. Testing full text extraction...")
        full_text = processor.extract_full_text(pdf_path)
        print(f"✓ Full text length: {len(full_text)} characters")
        print(f"✓ Preview: {full_text[:200]}...")
        
        print("\n" + "=" * 60)
        print("✅ PDF processing test completed successfully!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Custom PDF file provided
        pdf_file = sys.argv[1]
        test_with_custom_pdf(pdf_file)
    else:
        # Run standard tests
        test_pdf_processor()