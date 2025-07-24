#!/usr/bin/env python3
"""
Round 1B: Relevance Scoring Engine
Combines semantic and lexical similarity for persona-driven document intelligence
"""

import re
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter

# Import ML libraries
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from rapidfuzz import fuzz, process
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False

logger = logging.getLogger(__name__)

class RelevanceScorer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize with lightweight sentence transformer (<200MB)
        """
        self.model = None
        self.model_name = model_name
        
        # Scoring weights
        self.weights = {
            'semantic': 0.6,
            'lexical': 0.25,
            'title_match': 0.15
        }
        
        # Priority weights for search terms
        self.priority_weights = {
            'high_priority': 1.0,
            'medium_priority': 0.7,
            'low_priority': 0.4
        }
        
        # Initialize model if dependencies available
        if DEPENDENCIES_AVAILABLE:
            try:
                self._load_model()
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {e}")
                self.model = None
        else:
            logger.warning("Dependencies not available, using lexical scoring only")

    def _load_model(self):
        """Load and cache the sentence transformer model"""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def compute_semantic_similarity(self, section_text: str, persona_job_vector: str) -> float:
        """
        Compute embedding-based similarity using sentence transformers
        """
        if not self.model or not DEPENDENCIES_AVAILABLE:
            # Fallback to simple word overlap
            return self._compute_word_overlap(section_text, persona_job_vector)
        
        try:
            # Clean and prepare texts
            section_clean = self._clean_text(section_text)
            context_clean = self._clean_text(persona_job_vector)
            
            if not section_clean or not context_clean:
                return 0.0
            
            # Generate embeddings
            embeddings = self.model.encode([section_clean, context_clean])
            
            # Compute cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            # Ensure valid range [0, 1]
            return max(0.0, min(1.0, float(similarity)))
            
        except Exception as e:
            logger.warning(f"Semantic similarity computation failed: {e}")
            return self._compute_word_overlap(section_text, persona_job_vector)

    def compute_lexical_similarity(self, section_text: str, search_terms: Dict) -> float:
        """
        Compute keyword-based similarity using fuzzy matching
        """
        if not section_text or not search_terms:
            return 0.0
        
        section_lower = self._clean_text(section_text).lower()
        total_score = 0.0
        total_weight = 0.0
        
        # Score each priority level
        for priority, terms in search_terms.items():
            if not terms:
                continue
                
            priority_weight = self.priority_weights.get(priority, 0.5)
            priority_score = 0.0
            
            for term in terms:
                if not term or len(term.strip()) < 2:
                    continue
                    
                term_clean = term.strip().lower()
                
                # Direct match (highest score)
                if term_clean in section_lower:
                    priority_score += 1.0
                
                # Fuzzy match using rapidfuzz
                elif DEPENDENCIES_AVAILABLE:
                    try:
                        # Find best fuzzy match in section
                        words = section_lower.split()
                        if words:
                            best_match = process.extractOne(
                                term_clean, words, 
                                scorer=fuzz.WRatio,
                                score_cutoff=70
                            )
                            if best_match:
                                priority_score += best_match[1] / 100.0
                    except Exception:
                        # Fallback to simple substring matching
                        if any(word in term_clean or term_clean in word for word in section_lower.split()):
                            priority_score += 0.5
                else:
                    # Simple substring matching fallback
                    if any(word in term_clean or term_clean in word for word in section_lower.split()):
                        priority_score += 0.5
            
            # Normalize by number of terms
            if terms:
                priority_score = priority_score / len(terms)
                total_score += priority_score * priority_weight
                total_weight += priority_weight
        
        # Normalize final score
        if total_weight > 0:
            return min(1.0, total_score / total_weight)
        return 0.0

    def compute_title_match_score(self, section_title: str, search_terms: Dict) -> float:
        """
        Compute title-specific matching score
        """
        if not section_title:
            return 0.0
        
        title_lower = section_title.lower()
        max_score = 0.0
        
        # Check all search terms against title
        for priority, terms in search_terms.items():
            priority_weight = self.priority_weights.get(priority, 0.5)
            
            for term in terms:
                if not term or len(term.strip()) < 2:
                    continue
                    
                term_clean = term.strip().lower()
                
                # Direct match in title
                if term_clean in title_lower:
                    max_score = max(max_score, 1.0 * priority_weight)
                
                # Partial match
                elif any(word in term_clean or term_clean in word for word in title_lower.split()):
                    max_score = max(max_score, 0.6 * priority_weight)
        
        return min(1.0, max_score)

    def hybrid_score(self, section: Dict, persona_context: Dict) -> float:
        """
        Combine semantic + lexical + title scores with weights
        """
        try:
            section_text = section.get('text', '')
            section_title = section.get('title', '')
            search_terms = persona_context.get('search_terms', {})
            combined_text = persona_context.get('combined_text', '')
            
            # Compute individual scores
            semantic_score = self.compute_semantic_similarity(section_text, combined_text)
            lexical_score = self.compute_lexical_similarity(section_text, search_terms)
            title_score = self.compute_title_match_score(section_title, search_terms)
            
            # Combine with weights
            final_score = (
                self.weights['semantic'] * semantic_score +
                self.weights['lexical'] * lexical_score +
                self.weights['title_match'] * title_score
            )
            
            # Apply length penalty for very short sections
            if len(section_text) < 50:
                final_score *= 0.7
            
            # Apply bonus for longer, content-rich sections
            elif len(section_text) > 500:
                final_score *= 1.1
            
            return min(1.0, max(0.0, final_score))
            
        except Exception as e:
            logger.warning(f"Error computing hybrid score: {e}")
            return 0.0

    def rank_sections(self, all_sections: List[Dict], persona_context: Dict, top_k: int = 10) -> List[Dict]:
        """
        Rank all sections across documents by relevance score
        """
        if not all_sections:
            return []
        
        logger.info(f"Ranking {len(all_sections)} sections...")
        
        scored_sections = []
        
        for i, section in enumerate(all_sections):
            try:
                # Compute relevance score
                score = self.hybrid_score(section, persona_context)
                
                scored_sections.append({
                    'section': section,
                    'score': score,
                    'document': section.get('document', ''),
                    'page': section.get('page', 1),
                    'title': section.get('title', ''),
                    'text_length': len(section.get('text', '')),
                    'level': section.get('level', 1)
                })
                
                # Log progress
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(all_sections)} sections")
                    
            except Exception as e:
                logger.warning(f"Error scoring section {i}: {e}")
                continue
        
        # Sort by score (descending) and return top-k
        ranked = sorted(scored_sections, key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Ranking complete. Top score: {ranked[0]['score']:.3f}, Bottom score: {ranked[-1]['score']:.3f}")
        
        return ranked[:top_k]

    def analyze_subsections(self, ranked_sections: List[Dict], persona_context: Dict) -> List[Dict]:
        """
        Analyze subsections within top-ranked sections for granular insights
        """
        subsection_analysis = []
        
        for ranked_item in ranked_sections[:5]:  # Analyze top 5 sections in detail
            section = ranked_item['section']
            section_text = section.get('text', '')
            
            if len(section_text) < 100:  # Skip very short sections
                continue
            
            # Split into paragraphs
            paragraphs = [p.strip() for p in section_text.split('\n') if len(p.strip()) > 50]
            
            if len(paragraphs) <= 1:
                continue
            
            # Score each paragraph
            paragraph_scores = []
            for i, paragraph in enumerate(paragraphs):
                para_score = self.compute_lexical_similarity(paragraph, persona_context.get('search_terms', {}))
                
                if para_score > 0.1:  # Only include relevant paragraphs
                    paragraph_scores.append({
                        'paragraph_index': i,
                        'text': paragraph[:200] + '...' if len(paragraph) > 200 else paragraph,
                        'relevance_score': para_score,
                        'key_terms': self._extract_key_terms(paragraph, persona_context)
                    })
            
            if paragraph_scores:
                # Sort paragraphs by relevance
                paragraph_scores.sort(key=lambda x: x['relevance_score'], reverse=True)
                
                subsection_analysis.append({
                    'section_title': section.get('title', ''),
                    'document': section.get('document', ''),
                    'page': section.get('page', 1),
                    'overall_relevance': ranked_item['score'],
                    'relevant_paragraphs': paragraph_scores[:3],  # Top 3 paragraphs
                    'summary': self._generate_section_summary(section, persona_context)
                })
        
        return subsection_analysis

    def _extract_key_terms(self, text: str, persona_context: Dict) -> List[str]:
        """Extract key terms from text that match persona context"""
        search_terms = persona_context.get('search_terms', {})
        text_lower = text.lower()
        found_terms = []
        
        for priority, terms in search_terms.items():
            for term in terms:
                if term and term.lower() in text_lower:
                    found_terms.append(term)
        
        return list(set(found_terms))[:5]  # Top 5 key terms

    def _generate_section_summary(self, section: Dict, persona_context: Dict) -> str:
        """Generate a brief summary of why this section is relevant"""
        title = section.get('title', 'Section')
        key_terms = self._extract_key_terms(section.get('text', ''), persona_context)
        
        if key_terms:
            return f"{title} contains relevant information about: {', '.join(key_terms[:3])}"
        else:
            return f"{title} appears relevant to the user's requirements"

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for processing"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove very short words and common stop words
        words = text.split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        filtered_words = [word for word in words if len(word) > 2 and word.lower() not in stop_words]
        
        return ' '.join(filtered_words)

    def _compute_word_overlap(self, text1: str, text2: str) -> float:
        """Fallback similarity computation using word overlap"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(self._clean_text(text1).lower().split())
        words2 = set(self._clean_text(text2).lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def get_model_info(self) -> Dict:
        """Return information about the loaded model"""
        return {
            'model_name': self.model_name,
            'model_loaded': self.model is not None,
            'dependencies_available': DEPENDENCIES_AVAILABLE,
            'weights': self.weights
        }

def test_relevance_scorer():
    """Test relevance scorer functionality"""
    print("Testing Relevance Scorer...")
    
    scorer = RelevanceScorer()
    
    # Test model info
    model_info = scorer.get_model_info()
    print(f"✓ Model info: {model_info['model_name']}, Dependencies: {model_info['dependencies_available']}")
    
    # Test text cleaning
    dirty_text = "  This   is    a   test   with    extra    spaces   and short words like a, an, the  "
    clean_text = scorer._clean_text(dirty_text)
    print(f"✓ Text cleaning: '{dirty_text[:30]}...' -> '{clean_text[:30]}...'")
    
    # Test lexical similarity
    test_section_text = "This section discusses machine learning algorithms including neural networks, deep learning, and transformer architectures. We explore attention mechanisms and their applications."
    
    test_search_terms = {
        'high_priority': ['machine learning', 'neural networks', 'transformers'],
        'medium_priority': ['algorithms', 'deep learning'],
        'low_priority': ['applications']
    }
    
    lexical_score = scorer.compute_lexical_similarity(test_section_text, test_search_terms)
    print(f"✓ Lexical similarity score: {lexical_score:.3f}")
    
    # Test title matching
    test_title = "Deep Learning and Neural Networks"
    title_score = scorer.compute_title_match_score(test_title, test_search_terms)
    print(f"✓ Title match score: {title_score:.3f}")
    
    # Test semantic similarity (word overlap fallback)
    context_text = "machine learning artificial intelligence neural networks"
    semantic_score = scorer.compute_semantic_similarity(test_section_text, context_text)
    print(f"✓ Semantic similarity score: {semantic_score:.3f}")
    
    # Test hybrid scoring
    test_section = {
        'title': 'Machine Learning Fundamentals',
        'text': test_section_text,
        'page': 1,
        'level': 1
    }
    
    test_context = {
        'search_terms': test_search_terms,
        'combined_text': context_text
    }
    
    hybrid_score = scorer.hybrid_score(test_section, test_context)
    print(f"✓ Hybrid score: {hybrid_score:.3f}")
    
    # Test section ranking
    test_sections = [
        {
            'title': 'Introduction to AI',
            'text': 'Artificial intelligence and machine learning overview.',
            'page': 1,
            'document': 'ai_book.pdf'
        },
        {
            'title': 'Neural Network Architectures',
            'text': 'Deep learning using neural networks and transformers for natural language processing.',
            'page': 5,
            'document': 'ai_book.pdf'
        },
        {
            'title': 'Data Preprocessing',
            'text': 'Methods for cleaning and preparing data for analysis.',
            'page': 3,
            'document': 'data_book.pdf'
        }
    ]
    
    ranked_sections = scorer.rank_sections(test_sections, test_context, top_k=3)
    print(f"✓ Ranked {len(ranked_sections)} sections:")
    for i, item in enumerate(ranked_sections):
        print(f"  {i+1}. {item['title']} (score: {item['score']:.3f})")
    
    # Test key term extraction
    key_terms = scorer._extract_key_terms(test_section_text, test_context)
    print(f"✓ Extracted key terms: {key_terms}")
    
    print("Relevance Scorer test completed successfully!")
    return True

if __name__ == "__main__":
    test_relevance_scorer()