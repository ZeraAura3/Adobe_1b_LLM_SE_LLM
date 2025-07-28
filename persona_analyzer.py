"""
Persona and Job Analyzer Module for Round 1B
Analyzes persona and job-to-be-done to extract key requirements.
"""

import re
from typing import Dict, Any, List
try:
    import spacy
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
except ImportError:
    spacy = None
    nltk = None
    word_tokenize = None
    stopwords = None


class PersonaJobAnalyzer:
    """Analyzes persona and job-to-be-done to extract key requirements."""
    
    def __init__(self):
        self.persona_keywords = {}
        self.job_keywords = {}
        self.domain_keywords = self._build_domain_keywords()
        
        # Initialize NLP components
        try:
            self.nlp = spacy.load("en_core_web_sm") if spacy else None
            self.stop_words = set(stopwords.words('english')) if stopwords else set()
        except:
            self.nlp = None
            self.stop_words = set()
        
    def _build_domain_keywords(self) -> Dict[str, List[str]]:
        """Build domain-specific keyword mappings."""
        return {
            'academic_research': [
                'methodology', 'literature', 'review', 'research', 'study', 'analysis',
                'findings', 'results', 'discussion', 'hypothesis', 'experiment',
                'data', 'statistical', 'empirical', 'theoretical', 'framework'
            ],
            'business_analysis': [
                'revenue', 'profit', 'financial', 'market', 'strategy', 'competitive',
                'growth', 'investment', 'roi', 'performance', 'metrics', 'kpi',
                'trends', 'forecast', 'analysis', 'business', 'commercial'
            ],
            'education': [
                'concept', 'theory', 'principle', 'explanation', 'example',
                'practice', 'exercise', 'problem', 'solution', 'definition',
                'formula', 'equation', 'mechanism', 'process', 'steps'
            ],
            'technical': [
                'implementation', 'architecture', 'design', 'algorithm',
                'performance', 'optimization', 'system', 'technical',
                'specification', 'requirements', 'documentation'
            ]
        }
    
    def analyze_persona(self, persona_description: str) -> Dict[str, Any]:
        """Extract key information from persona description."""
        persona_analysis = {
            'role': '',
            'expertise_areas': [],
            'focus_keywords': [],
            'domain': '',
            'experience_level': ''
        }
        
        text_lower = persona_description.lower()
        
        # Extract role
        role_patterns = [
            r'(phd researcher|researcher|student|analyst|manager|developer|engineer|scientist)',
            r'(undergraduate|graduate|professor|teacher|instructor)',
            r'(investment analyst|business analyst|data analyst)',
            r'(salesperson|entrepreneur|journalist)'
        ]
        
        for pattern in role_patterns:
            match = re.search(pattern, text_lower)
            if match:
                persona_analysis['role'] = match.group(1)
                break
        
        # Determine domain based on context
        for domain, keywords in self.domain_keywords.items():
            domain_score = sum(1 for keyword in keywords if keyword in text_lower)
            if domain_score >= 2:  # Threshold for domain detection
                persona_analysis['domain'] = domain
                persona_analysis['focus_keywords'].extend(keywords[:10])  # Top keywords
                break
        
        # Extract expertise areas using NLP
        if self.nlp:
            try:
                doc = self.nlp(persona_description)
                # Extract noun phrases as potential expertise areas
                expertise_areas = [chunk.text.lower() for chunk in doc.noun_chunks 
                                 if len(chunk.text.split()) <= 3]
                persona_analysis['expertise_areas'] = expertise_areas[:5]
            except:
                pass
        
        # Experience level detection
        if any(term in text_lower for term in ['phd', 'senior', 'lead', 'principal']):
            persona_analysis['experience_level'] = 'expert'
        elif any(term in text_lower for term in ['undergraduate', 'junior', 'entry']):
            persona_analysis['experience_level'] = 'beginner'
        else:
            persona_analysis['experience_level'] = 'intermediate'
        
        return persona_analysis
    
    def analyze_job(self, job_description: str) -> Dict[str, Any]:
        """Extract key requirements from job-to-be-done."""
        job_analysis = {
            'task_type': '',
            'required_info': [],
            'output_format': '',
            'key_concepts': [],
            'priority_keywords': []
        }
        
        text_lower = job_description.lower()
        
        # Detect task type
        if any(term in text_lower for term in ['review', 'survey', 'overview']):
            job_analysis['task_type'] = 'literature_review'
        elif any(term in text_lower for term in ['analyze', 'analysis', 'examine']):
            job_analysis['task_type'] = 'analysis'
        elif any(term in text_lower for term in ['summarize', 'summary']):
            job_analysis['task_type'] = 'summarization'
        elif any(term in text_lower for term in ['prepare', 'study', 'learn']):
            job_analysis['task_type'] = 'preparation'
        elif any(term in text_lower for term in ['identify', 'find', 'extract']):
            job_analysis['task_type'] = 'information_extraction'
        
        # Extract key concepts and requirements
        important_phrases = []
        if self.nlp:
            try:
                doc = self.nlp(job_description)
                # Extract key noun phrases
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) <= 4:
                        important_phrases.append(chunk.text.lower())
                
                # Extract entities
                entities = [ent.text.lower() for ent in doc.ents 
                           if ent.label_ in ['ORG', 'PRODUCT', 'EVENT', 'WORK_OF_ART']]
                important_phrases.extend(entities)
            except:
                pass
        
        job_analysis['key_concepts'] = important_phrases[:10]
        
        # Extract priority keywords based on action verbs and important nouns
        priority_words = []
        if word_tokenize and nltk:
            try:
                words = word_tokenize(text_lower)
                pos_tags = nltk.pos_tag(words)
                
                for word, pos in pos_tags:
                    if pos.startswith('VB') or pos.startswith('NN'):  # Verbs and nouns
                        if word not in self.stop_words and len(word) > 3:
                            priority_words.append(word)
            except:
                # Fallback to simple word extraction
                words = text_lower.split()
                priority_words = [w for w in words if len(w) > 3 and w not in self.stop_words]
        
        job_analysis['priority_keywords'] = list(set(priority_words))[:15]
        
        return job_analysis
    
    def combine_requirements(self, persona_analysis: Dict, job_analysis: Dict) -> Dict[str, Any]:
        """Combine persona and job requirements for relevance scoring."""
        combined = {
            'all_keywords': [],
            'domain_focus': persona_analysis.get('domain', ''),
            'task_priority': job_analysis.get('task_type', ''),
            'expertise_level': persona_analysis.get('experience_level', 'intermediate'),
            'weighted_keywords': {}
        }
        
        # Combine all keywords
        all_keywords = []
        all_keywords.extend(persona_analysis.get('focus_keywords', []))
        all_keywords.extend(job_analysis.get('priority_keywords', []))
        all_keywords.extend(persona_analysis.get('expertise_areas', []))
        all_keywords.extend(job_analysis.get('key_concepts', []))
        
        # Create weighted keywords (higher weight for job-specific terms)
        keyword_weights = {}
        for keyword in persona_analysis.get('focus_keywords', []):
            keyword_weights[keyword] = 1.0
        
        for keyword in job_analysis.get('priority_keywords', []):
            keyword_weights[keyword] = 2.0  # Higher weight for job requirements
        
        for concept in job_analysis.get('key_concepts', []):
            keyword_weights[concept] = 1.5
        
        combined['all_keywords'] = list(set(all_keywords))
        combined['weighted_keywords'] = keyword_weights
        
        return combined
