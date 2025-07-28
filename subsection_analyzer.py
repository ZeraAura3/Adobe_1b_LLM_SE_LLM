"""
Sub-section Analyzer Module for Round 1B
Performs granular sub-section analysis on top-ranked sections.
"""

import re
from typing import Dict, Any, List
try:
    from nltk.tokenize import sent_tokenize
except ImportError:
    def sent_tokenize(text):
        """Fallback sentence tokenizer"""
        return re.split(r'[.!?]+', text)


class SubSectionAnalyzer:
    """Performs granular sub-section analysis on top-ranked sections."""
    
    def __init__(self):
        self.sentence_tokenizer = sent_tokenize
        
    def analyze_subsections(self, documents: Dict[str, Any], top_sections: List[Dict], 
                           requirements: Dict[str, Any], max_subsections: int = 20) -> List[Dict]:
        """Extract and analyze sub-sections from top-ranked sections."""
        print(f"🔍 Analyzing sub-sections from top {len(top_sections)} sections...")
        
        subsections = []
        
        for section_info in top_sections[:5]:  # Analyze top 5 sections for sub-sections
            doc_name = section_info['document']
            section_title = section_info['section_title']
            page_num = section_info['page_number']
            
            # Find the actual section content
            section_content = self._find_section_content(documents, doc_name, section_title, page_num)
            
            if section_content:
                # Extract sub-sections using multiple strategies
                extracted_subsections = self._extract_subsections(
                    content=section_content,
                    doc_name=doc_name,
                    parent_section=section_title,
                    page_num=page_num,
                    requirements=requirements
                )
                
                subsections.extend(extracted_subsections)
        
        # Score and rank sub-sections
        scored_subsections = self._score_subsections(subsections, requirements)
        
        # Return top sub-sections
        return scored_subsections[:max_subsections]
    
    def _find_section_content(self, documents: Dict, doc_name: str, section_title: str, 
                            page_num: int) -> str:
        """Find the content of a specific section."""
        if doc_name not in documents:
            return ""
        
        doc_data = documents[doc_name]
        for section in doc_data['sections']:
            if (section['title'] == section_title and 
                section['page_number'] == page_num):
                return section['content']
        
        return ""
    
    def _extract_subsections(self, content: str, doc_name: str, parent_section: str,
                           page_num: int, requirements: Dict) -> List[Dict]:
        """Extract sub-sections using multiple strategies."""
        subsections = []
        
        # Strategy 1: Paragraph-based sub-sections
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.split()) >= 20:  # Minimum length for meaningful content
                subsections.append({
                    'document': doc_name,
                    'parent_section': parent_section,
                    'page_number': page_num,
                    'subsection_type': 'paragraph',
                    'subsection_index': i,
                    'content': paragraph,
                    'word_count': len(paragraph.split()),
                    'extraction_method': 'paragraph_split'
                })
        
        # Strategy 2: Sentence-based sub-sections for shorter content
        if len(subsections) < 2:  # If few paragraphs, try sentence grouping
            sentences = self.sentence_tokenizer(content)
            
            # Group sentences into meaningful chunks
            sentence_groups = self._group_sentences(sentences, min_words=30, max_words=150)
            
            for i, group in enumerate(sentence_groups):
                if len(group.split()) >= 20:
                    subsections.append({
                        'document': doc_name,
                        'parent_section': parent_section,
                        'page_number': page_num,
                        'subsection_type': 'sentence_group',
                        'subsection_index': i,
                        'content': group,
                        'word_count': len(group.split()),
                        'extraction_method': 'sentence_grouping'
                    })
        
        # Strategy 3: Key phrase extraction for very dense content
        if len(content.split()) > 500:  # For very long sections
            key_passages = self._extract_key_passages(content, requirements, max_passages=3)
            
            for i, passage in enumerate(key_passages):
                subsections.append({
                    'document': doc_name,
                    'parent_section': parent_section,
                    'page_number': page_num,
                    'subsection_type': 'key_passage',
                    'subsection_index': i,
                    'content': passage,
                    'word_count': len(passage.split()),
                    'extraction_method': 'key_phrase_extraction'
                })
        
        return subsections
    
    def _group_sentences(self, sentences: List[str], min_words: int = 30, 
                        max_words: int = 150) -> List[str]:
        """Group sentences into coherent chunks."""
        groups = []
        current_group = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_words = len(sentence.split())
            
            if current_word_count + sentence_words <= max_words:
                current_group.append(sentence)
                current_word_count += sentence_words
            else:
                if current_word_count >= min_words:
                    groups.append(' '.join(current_group))
                current_group = [sentence]
                current_word_count = sentence_words
        
        # Add final group
        if current_group and current_word_count >= min_words:
            groups.append(' '.join(current_group))
        
        return groups
    
    def _extract_key_passages(self, content: str, requirements: Dict, 
                            max_passages: int = 3) -> List[str]:
        """Extract key passages based on keyword density."""
        sentences = self.sentence_tokenizer(content)
        keywords = requirements.get('all_keywords', [])
        
        # Score sentences based on keyword presence
        sentence_scores = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for keyword in keywords if keyword in sentence_lower)
            sentence_scores.append((sentence, score))
        
        # Sort by score and select top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Group high-scoring adjacent sentences
        passages = []
        used_indices = set()
        
        for i, (sentence, score) in enumerate(sentence_scores[:max_passages * 2]):
            if i in used_indices or score == 0:
                continue
                
            # Find sentence index in original list
            try:
                orig_index = sentences.index(sentence)
            except ValueError:
                continue
            
            # Create passage with adjacent sentences
            start_idx = max(0, orig_index - 1)
            end_idx = min(len(sentences), orig_index + 2)
            
            passage_sentences = sentences[start_idx:end_idx]
            passage = ' '.join(passage_sentences)
            
            if len(passage.split()) >= 20:
                passages.append(passage)
                used_indices.update(range(start_idx, end_idx))
            
            if len(passages) >= max_passages:
                break
        
        return passages
    
    def _score_subsections(self, subsections: List[Dict], 
                          requirements: Dict) -> List[Dict]:
        """Score and rank sub-sections based on relevance."""
        keywords = requirements.get('all_keywords', [])
        weighted_keywords = requirements.get('weighted_keywords', {})
        
        for subsection in subsections:
            content_lower = subsection['content'].lower()
            
            # Base score: keyword density
            base_score = 0
            for keyword in keywords:
                if keyword in content_lower:
                    weight = weighted_keywords.get(keyword, 1.0)
                    # Count occurrences with diminishing returns
                    occurrences = content_lower.count(keyword)
                    score_contribution = weight * min(occurrences, 3)  # Cap at 3 occurrences
                    base_score += score_contribution
            
            # Normalize by content length
            word_count = subsection['word_count']
            normalized_score = base_score / max(word_count / 100, 1)  # Per 100 words
            
            # Bonus for optimal length
            length_bonus = 1.0
            if 50 <= word_count <= 200:
                length_bonus = 1.2
            elif 30 <= word_count < 50 or 200 < word_count <= 300:
                length_bonus = 1.1
            
            # Bonus for certain extraction methods
            method_bonus = 1.0
            if subsection['extraction_method'] == 'key_phrase_extraction':
                method_bonus = 1.3
            elif subsection['extraction_method'] == 'paragraph_split':
                method_bonus = 1.1
            
            final_score = normalized_score * length_bonus * method_bonus
            subsection['relevance_score'] = round(final_score, 4)
            
            # Add refined text (cleaned up version)
            subsection['refined_text'] = self._refine_text(subsection['content'])
        
        # Sort by relevance score
        subsections.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Add ranks
        for rank, subsection in enumerate(subsections, 1):
            subsection['importance_rank'] = rank
        
        return subsections
    
    def _refine_text(self, text: str) -> str:
        """Clean and refine text for better readability."""
        # Remove excessive whitespace
        refined = re.sub(r'\s+', ' ', text)
        
        # Remove incomplete sentences at the beginning/end
        sentences = self.sentence_tokenizer(refined)
        if len(sentences) > 1:
            # Keep complete sentences
            complete_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence[-1] in '.!?':
                    complete_sentences.append(sentence)
                elif len(sentence.split()) > 10:  # Keep long fragments
                    complete_sentences.append(sentence)
            
            if complete_sentences:
                refined = ' '.join(complete_sentences)
        
        # Limit length
        words = refined.split()
        if len(words) > 200:
            refined = ' '.join(words[:200]) + '...'
        
        return refined.strip()
