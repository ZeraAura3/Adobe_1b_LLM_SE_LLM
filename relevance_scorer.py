"""
Relevance Scorer Module for Round 1B
Scores document sections based on persona and job requirements.
"""

from typing import Dict, Any, List
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    np = None
    TfidfVectorizer = None
    cosine_similarity = None


class RelevanceScorer:
    """Scores document sections based on persona and job requirements."""
    
    def __init__(self):
        if TfidfVectorizer:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,  # Limit features for CPU efficiency
                stop_words='english',
                ngram_range=(1, 2),  # Unigrams and bigrams
                min_df=1,
                max_df=0.95
            )
        else:
            self.tfidf_vectorizer = None
        
        self.section_vectors = None
        self.requirement_vector = None
        
    def prepare_scoring(self, documents: Dict[str, Any], requirements: Dict[str, Any]):
        """Prepare TF-IDF vectors for scoring."""
        # Collect all section texts
        section_texts = []
        self.section_metadata = []
        
        for doc_name, doc_data in documents.items():
            for section in doc_data['sections']:
                section_texts.append(section['content'])
                self.section_metadata.append({
                    'document': doc_name,
                    'title': section['title'],
                    'page_number': section['page_number'],
                    'word_count': section['word_count'],
                    'section_index': len(self.section_metadata)
                })
        
        if not section_texts:
            print("❌ No sections found for scoring")
            return
        
        # Create TF-IDF vectors if available
        if self.tfidf_vectorizer:
            print(f"🔢 Creating TF-IDF vectors for {len(section_texts)} sections...")
            all_texts = section_texts + [' '.join(requirements['all_keywords'])]
            
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
                self.section_vectors = tfidf_matrix[:-1]  # All but last (requirement vector)
                self.requirement_vector = tfidf_matrix[-1]  # Last one is requirements
                print(f"✅ TF-IDF preparation complete: {self.section_vectors.shape}")
            except Exception as e:
                print(f"❌ Error in TF-IDF preparation: {e}")
                self._fallback_scoring_preparation(section_texts, requirements)
        else:
            self._fallback_scoring_preparation(section_texts, requirements)
    
    def _fallback_scoring_preparation(self, section_texts: List[str], requirements: Dict[str, Any]):
        """Fallback scoring when sklearn is not available."""
        print("⚠️  Using fallback keyword-based scoring")
        self.section_texts = section_texts
        self.requirements = requirements
    
    def calculate_relevance_scores(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate relevance scores for all sections."""
        if self.section_vectors is not None and self.requirement_vector is not None:
            return self._calculate_tfidf_scores(requirements)
        else:
            return self._calculate_fallback_scores(requirements)
    
    def _calculate_tfidf_scores(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate scores using TF-IDF and cosine similarity."""
        print("📊 Calculating TF-IDF relevance scores...")
        
        # Calculate cosine similarity
        similarity_scores = cosine_similarity(self.section_vectors, self.requirement_vector).flatten()
        
        # Calculate additional scoring factors
        scored_sections = []
        
        for i, (score, metadata) in enumerate(zip(similarity_scores, self.section_metadata)):
            section_score = self._calculate_comprehensive_score(
                base_score=score,
                section_metadata=metadata,
                requirements=requirements
            )
            
            scored_sections.append({
                **metadata,
                'base_similarity': float(score),
                'comprehensive_score': section_score,
                'relevance_factors': self._get_relevance_factors(metadata, requirements)
            })
        
        # Sort by comprehensive score
        scored_sections.sort(key=lambda x: x['comprehensive_score'], reverse=True)
        
        # Add importance ranks
        for rank, section in enumerate(scored_sections, 1):
            section['importance_rank'] = rank
        
        print(f"✅ Scored {len(scored_sections)} sections")
        return scored_sections
    
    def _calculate_fallback_scores(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback scoring using simple keyword matching."""
        print("📊 Calculating keyword-based relevance scores...")
        
        scored_sections = []
        keywords = requirements.get('all_keywords', [])
        
        for i, metadata in enumerate(self.section_metadata):
            section_text = self.section_texts[i].lower()
            
            # Count keyword matches
            keyword_score = sum(1 for keyword in keywords if keyword in section_text)
            
            # Normalize by text length
            normalized_score = keyword_score / max(len(section_text.split()) / 100, 1)
            
            section_score = self._calculate_comprehensive_score(
                base_score=normalized_score,
                section_metadata=metadata,
                requirements=requirements
            )
            
            scored_sections.append({
                **metadata,
                'base_similarity': normalized_score,
                'comprehensive_score': section_score,
                'relevance_factors': self._get_relevance_factors(metadata, requirements)
            })
        
        # Sort by comprehensive score
        scored_sections.sort(key=lambda x: x['comprehensive_score'], reverse=True)
        
        # Add importance ranks
        for rank, section in enumerate(scored_sections, 1):
            section['importance_rank'] = rank
        
        print(f"✅ Scored {len(scored_sections)} sections")
        return scored_sections
    
    def _calculate_comprehensive_score(self, base_score: float, section_metadata: Dict, 
                                     requirements: Dict) -> float:
        """Calculate comprehensive relevance score with multiple factors."""
        score = base_score
        
        # Length bonus (moderate length preferred)
        word_count = section_metadata['word_count']
        if 100 <= word_count <= 500:
            score *= 1.1
        elif 50 <= word_count < 100 or 500 < word_count <= 1000:
            score *= 1.05
        elif word_count < 50:
            score *= 0.8
        
        # Title relevance bonus
        title_lower = section_metadata['title'].lower()
        keyword_matches = sum(1 for keyword in requirements.get('all_keywords', []) 
                            if keyword in title_lower)
        if keyword_matches > 0:
            score *= (1 + 0.1 * keyword_matches)
        
        # Domain-specific bonuses
        domain = requirements.get('domain_focus', '')
        if domain == 'academic_research':
            academic_terms = ['methodology', 'results', 'discussion', 'analysis', 'findings']
            if any(term in title_lower for term in academic_terms):
                score *= 1.15
        elif domain == 'business_analysis':
            business_terms = ['financial', 'revenue', 'performance', 'market', 'strategy']
            if any(term in title_lower for term in business_terms):
                score *= 1.15
        elif domain == 'education':
            edu_terms = ['concept', 'theory', 'principle', 'example', 'problem']
            if any(term in title_lower for term in edu_terms):
                score *= 1.15
        
        return score
    
    def _get_relevance_factors(self, metadata: Dict, requirements: Dict) -> List[str]:
        """Get human-readable relevance factors."""
        factors = []
        
        title_lower = metadata['title'].lower()
        
        # Check for keyword matches
        keyword_matches = [kw for kw in requirements.get('all_keywords', []) 
                          if kw in title_lower]
        if keyword_matches:
            factors.append(f"Title matches: {', '.join(keyword_matches[:3])}")
        
        # Check for domain relevance
        domain = requirements.get('domain_focus', '')
        domain_keywords = {
            'academic_research': ['methodology', 'results', 'analysis'],
            'business_analysis': ['financial', 'performance', 'market'],
            'education': ['concept', 'theory', 'example']
        }
        
        if domain in domain_keywords:
            matches = [term for term in domain_keywords[domain] if term in title_lower]
            if matches:
                factors.append(f"Domain relevance: {', '.join(matches)}")
        
        # Section length factor
        word_count = metadata['word_count']
        if 100 <= word_count <= 500:
            factors.append("Optimal length")
        elif word_count > 500:
            factors.append("Comprehensive content")
        
        return factors[:3]  # Limit to top 3 factors
    
    def extract_top_sections(self, scored_sections: List[Dict], top_n: int = 10) -> List[Dict]:
        """Extract top N sections with their metadata."""
        top_sections = scored_sections[:top_n]
        
        result = []
        for section in top_sections:
            result.append({
                'document': section['document'],
                'page_number': section['page_number'],
                'section_title': section['title'],
                'importance_rank': section['importance_rank'],
                'relevance_score': round(section['comprehensive_score'], 4),
                'relevance_factors': section['relevance_factors']
            })
        
        return result
