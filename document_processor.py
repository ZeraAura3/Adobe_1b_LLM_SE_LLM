"""
Document Processor Module for Round 1B
Handles PDF document processing and text extraction.
"""

import re
from pathlib import Path
from typing import List, Dict, Any
import pdfplumber


class DocumentProcessor:
    """Handles PDF document processing and text extraction."""
    
    def __init__(self):
        self.documents = {}
        
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text and structure from PDF."""
        try:
            doc_data = {
                'file_path': pdf_path,
                'file_name': Path(pdf_path).name,
                'pages': [],
                'total_pages': 0,
                'sections': [],
                'full_text': ''
            }
            
            # Method 1: pdfplumber for better structure detection
            with pdfplumber.open(pdf_path) as pdf:
                doc_data['total_pages'] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    
                    page_data = {
                        'page_number': page_num,
                        'text': page_text,
                        'char_count': len(page_text),
                        'sections': self._extract_sections_from_page(page_text, page_num)
                    }
                    
                    doc_data['pages'].append(page_data)
                    doc_data['full_text'] += page_text + "\n"
                    doc_data['sections'].extend(page_data['sections'])
            
            return doc_data
            
        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {e}")
            return None
    
    def _extract_sections_from_page(self, text: str, page_num: int) -> List[Dict]:
        """Extract sections from page text using heuristics."""
        sections = []
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect potential section headers (various patterns)
            if self._is_section_header(line):
                # Save previous section
                if current_section and current_content:
                    sections.append({
                        'title': current_section,
                        'content': '\n'.join(current_content),
                        'page_number': page_num,
                        'word_count': len(' '.join(current_content).split()),
                        'char_count': len('\n'.join(current_content))
                    })
                
                # Start new section
                current_section = line
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
        
        # Add final section
        if current_section and current_content:
            sections.append({
                'title': current_section,
                'content': '\n'.join(current_content),
                'page_number': page_num,
                'word_count': len(' '.join(current_content).split()),
                'char_count': len('\n'.join(current_content))
            })
        
        return sections
    
    def _is_section_header(self, line: str) -> bool:
        """Heuristic to detect section headers."""
        line = line.strip()
        
        # Empty or too short
        if len(line) < 3 or len(line) > 200:
            return False
        
        # Numbered sections (1., 1.1, I., A., etc.)
        if re.match(r'^[0-9]+\.', line) or re.match(r'^[0-9]+\.[0-9]+', line):
            return True
        if re.match(r'^[IVX]+\.', line) or re.match(r'^[A-Z]\.', line):
            return True
            
        # All caps (but not too long)
        if line.isupper() and len(line.split()) <= 10:
            return True
            
        # Title case with limited words
        words = line.split()
        if (len(words) <= 8 and 
            all(word[0].isupper() or word.lower() in ['a', 'an', 'the', 'of', 'in', 'on', 'for', 'with'] 
                for word in words if word)):
            return True
        
        # Common section keywords
        section_keywords = ['introduction', 'methodology', 'results', 'discussion', 'conclusion',
                          'abstract', 'background', 'literature', 'methods', 'analysis',
                          'summary', 'references', 'appendix', 'overview', 'approach']
        
        if any(keyword in line.lower() for keyword in section_keywords):
            return True
            
        return False
    
    def process_documents(self, pdf_paths: List[str]) -> Dict[str, Any]:
        """Process multiple PDF documents."""
        print(f"📚 Processing {len(pdf_paths)} documents...")
        
        processed_docs = {}
        for pdf_path in pdf_paths:
            print(f"📖 Processing: {Path(pdf_path).name}")
            doc_data = self.extract_text_from_pdf(pdf_path)
            if doc_data:
                processed_docs[Path(pdf_path).name] = doc_data
                print(f"   ✅ Extracted {doc_data['total_pages']} pages, {len(doc_data['sections'])} sections")
            else:
                print(f"   ❌ Failed to process {pdf_path}")
        
        self.documents = processed_docs
        return processed_docs
