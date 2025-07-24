#!/usr/bin/env python3
"""
Round 1B: Output Generator
Generates final JSON output matching Round 1B requirements
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class OutputGenerator:
    def __init__(self):
        self.output_schema = {
            "metadata": {
                "input_documents": [],
                "persona": "",
                "job_to_be_done": "",
                "processing_timestamp": "",
                "total_sections_analyzed": 0,
                "top_sections_selected": 0,
                "model_info": {}
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }

    def generate_json_output(self, ranked_sections: List[Dict], subsection_analysis: List[Dict], 
                           metadata: Dict, persona_context: Dict) -> Dict:
        """
        Generate final JSON output matching required format
        """
        try:
            # Create base output structure
            output = {
                "metadata": self._generate_metadata(ranked_sections, metadata, persona_context),
                "extracted_sections": self._format_extracted_sections(ranked_sections),
                "subsection_analysis": subsection_analysis
            }
            
            # Validate output structure
            self._validate_output(output)
            
            logger.info(f"Generated output with {len(output['extracted_sections'])} sections and {len(output['subsection_analysis'])} subsection analyses")
            
            return output
            
        except Exception as e:
            logger.error(f"Error generating output: {e}")
            return self._generate_fallback_output(metadata, persona_context)

    def _generate_metadata(self, ranked_sections: List[Dict], metadata: Dict, persona_context: Dict) -> Dict:
        """Generate metadata section"""
        input_documents = list(set([
            section.get('document', 'unknown') 
            for section in ranked_sections 
            if 'section' in section
        ]))
        
        return {
            "input_documents": input_documents,
            "persona": metadata.get('persona', ''),
            "job_to_be_done": metadata.get('job', ''),
            "processing_timestamp": datetime.now().isoformat(),
            "total_sections_analyzed": metadata.get('total_sections', len(ranked_sections)),
            "top_sections_selected": len(ranked_sections),
            "model_info": metadata.get('model_info', {}),
            "persona_analysis": {
                "extracted_role": persona_context.get('persona', {}).get('role', ''),
                "identified_domain": persona_context.get('persona', {}).get('domain', ''),
                "expertise_level": persona_context.get('persona', {}).get('expertise_level', ''),
                "key_search_terms": len([
                    term for terms in persona_context.get('search_terms', {}).values() 
                    for term in terms
                ])
            },
            "job_analysis": {
                "primary_job_type": persona_context.get('job', {}).get('job_type', ''),
                "main_goals": persona_context.get('job', {}).get('main_goals', []),
                "deliverable_type": persona_context.get('job', {}).get('deliverable_type', ''),
                "urgency": persona_context.get('job', {}).get('urgency', 'normal')
            }
        }

    def _format_extracted_sections(self, ranked_sections: List[Dict]) -> List[Dict]:
        """Format extracted sections for output"""
        formatted_sections = []
        
        for i, ranked_item in enumerate(ranked_sections):
            section = ranked_item.get('section', {})
            
            formatted_section = {
                "rank": i + 1,
                "relevance_score": round(ranked_item.get('score', 0.0), 4),
                "document_name": section.get('document', 'unknown'),
                "section_title": section.get('title', f'Section {i+1}'),
                "page_number": section.get('page', 1),
                "page_range": {
                    "start": section.get('start_page', section.get('page', 1)),
                    "end": section.get('end_page', section.get('page', 1))
                },
                "heading_level": section.get('level', 1),
                "content": {
                    "text": section.get('text', ''),
                    "word_count": len(section.get('text', '').split()),
                    "character_count": len(section.get('text', ''))
                },
                "relevance_indicators": {
                    "contains_high_priority_terms": self._check_priority_terms(section.get('text', ''), 'high'),
                    "contains_medium_priority_terms": self._check_priority_terms(section.get('text', ''), 'medium'),
                    "title_relevance": ranked_item.get('score', 0.0) > 0.5
                },
                "extraction_metadata": {
                    "text_length": len(section.get('text', '')),
                    "pdf_path": section.get('pdf_path', ''),
                    "extraction_confidence": min(1.0, ranked_item.get('score', 0.0) + 0.2)
                }
            }
            
            formatted_sections.append(formatted_section)
        
        return formatted_sections

    def _check_priority_terms(self, text: str, priority: str) -> bool:
        """Check if text contains priority terms (placeholder for now)"""
        # This would normally check against the persona context search terms
        # For now, return a simple heuristic
        return len(text) > 100 and priority in ['high', 'medium']

    def _validate_output(self, output: Dict) -> bool:
        """Validate output structure matches requirements"""
        required_keys = ['metadata', 'extracted_sections', 'subsection_analysis']
        
        for key in required_keys:
            if key not in output:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate metadata
        metadata = output['metadata']
        required_metadata_keys = [
            'input_documents', 'persona', 'job_to_be_done', 
            'processing_timestamp', 'total_sections_analyzed'
        ]
        
        for key in required_metadata_keys:
            if key not in metadata:
                logger.warning(f"Missing metadata key: {key}")
        
        # Validate extracted sections structure
        for i, section in enumerate(output['extracted_sections']):
            required_section_keys = ['rank', 'relevance_score', 'document_name', 'section_title']
            for key in required_section_keys:
                if key not in section:
                    logger.warning(f"Section {i} missing key: {key}")
        
        return True

    def _generate_fallback_output(self, metadata: Dict, persona_context: Dict) -> Dict:
        """Generate minimal fallback output if main generation fails"""
        return {
            "metadata": {
                "input_documents": metadata.get('documents', []),
                "persona": metadata.get('persona', ''),
                "job_to_be_done": metadata.get('job', ''),
                "processing_timestamp": datetime.now().isoformat(),
                "total_sections_analyzed": 0,
                "top_sections_selected": 0,
                "error": "Fallback output generated due to processing error"
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }

    def save_output(self, output: Dict, output_path: str) -> bool:
        """Save output to JSON file"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Output saved to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving output to {output_path}: {e}")
            return False

    def save_debug_info(self, ranked_sections: List[Dict], output_dir: str) -> bool:
        """Save debug information for analysis"""
        try:
            debug_info = {
                "ranking_details": [
                    {
                        "rank": i + 1,
                        "score": item.get('score', 0.0),
                        "title": item.get('title', ''),
                        "document": item.get('document', ''),
                        "text_length": item.get('text_length', 0)
                    }
                    for i, item in enumerate(ranked_sections)
                ],
                "score_distribution": self._analyze_score_distribution(ranked_sections),
                "document_coverage": self._analyze_document_coverage(ranked_sections)
            }
            
            debug_path = Path(output_dir) / "debug_info.json"
            with open(debug_path, 'w', encoding='utf-8') as f:
                json.dump(debug_info, f, indent=2)
            
            logger.info(f"Debug info saved to: {debug_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving debug info: {e}")
            return False

    def _analyze_score_distribution(self, ranked_sections: List[Dict]) -> Dict:
        """Analyze the distribution of relevance scores"""
        scores = [item.get('score', 0.0) for item in ranked_sections]
        
        if not scores:
            return {}
        
        return {
            "min_score": min(scores),
            "max_score": max(scores),
            "mean_score": sum(scores) / len(scores),
            "score_range": max(scores) - min(scores),
            "high_relevance_count": len([s for s in scores if s > 0.7]),
            "medium_relevance_count": len([s for s in scores if 0.3 < s <= 0.7]),
            "low_relevance_count": len([s for s in scores if s <= 0.3])
        }

    def _analyze_document_coverage(self, ranked_sections: List[Dict]) -> Dict:
        """Analyze coverage across input documents"""
        doc_counts = {}
        for item in ranked_sections:
            doc = item.get('document', 'unknown')
            doc_counts[doc] = doc_counts.get(doc, 0) + 1
        
        return {
            "documents_represented": len(doc_counts),
            "sections_per_document": doc_counts,
            "most_relevant_document": max(doc_counts, key=doc_counts.get) if doc_counts else None
        }

    def generate_summary_report(self, output: Dict, output_dir: str) -> bool:
        """Generate a human-readable summary report"""
        try:
            sections = output.get('extracted_sections', [])
            metadata = output.get('metadata', {})
            
            report_lines = [
                "=== PERSONA-DRIVEN DOCUMENT INTELLIGENCE REPORT ===",
                f"Generated: {metadata.get('processing_timestamp', 'Unknown')}",
                "",
                f"PERSONA: {metadata.get('persona', 'Not specified')}",
                f"JOB TO BE DONE: {metadata.get('job_to_be_done', 'Not specified')}",
                "",
                f"DOCUMENTS ANALYZED: {', '.join(metadata.get('input_documents', []))}",
                f"TOTAL SECTIONS FOUND: {metadata.get('total_sections_analyzed', 0)}",
                f"TOP RELEVANT SECTIONS: {len(sections)}",
                "",
                "=== TOP RELEVANT SECTIONS ===",
            ]
            
            for section in sections[:5]:  # Top 5 sections
                report_lines.extend([
                    f"\n{section.get('rank', 0)}. {section.get('section_title', 'Untitled')}",
                    f"   Document: {section.get('document_name', 'Unknown')}",
                    f"   Page: {section.get('page_number', 'Unknown')}",
                    f"   Relevance Score: {section.get('relevance_score', 0.0):.3f}",
                    f"   Word Count: {section.get('content', {}).get('word_count', 0)}",
                ])
            
            report_path = Path(output_dir) / "summary_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            logger.info(f"Summary report saved to: {report_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return False

def test_output_generator():
    """Test output generator functionality"""
    print("Testing Output Generator...")
    
    generator = OutputGenerator()
    
    # Test data
    test_ranked_sections = [
        {
            'section': {
                'title': 'Machine Learning Overview',
                'text': 'This section provides an overview of machine learning techniques including supervised and unsupervised learning methods.',
                'page': 1,
                'level': 1,
                'document': 'ml_textbook.pdf',
                'start_page': 1,
                'end_page': 3
            },
            'score': 0.85,
            'document': 'ml_textbook.pdf',
            'page': 1,
            'title': 'Machine Learning Overview',
            'text_length': 120
        },
        {
            'section': {
                'title': 'Neural Networks',
                'text': 'Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes that process information.',
                'page': 5,
                'level': 2,
                'document': 'neural_nets.pdf',
                'start_page': 5,
                'end_page': 7
            },
            'score': 0.72,
            'document': 'neural_nets.pdf',
            'page': 5,
            'title': 'Neural Networks',
            'text_length': 140
        }
    ]
    
    test_subsection_analysis = [
        {
            'section_title': 'Machine Learning Overview',
            'document': 'ml_textbook.pdf',
            'page': 1,
            'overall_relevance': 0.85,
            'relevant_paragraphs': [
                {
                    'paragraph_index': 0,
                    'text': 'This paragraph discusses supervised learning methods...',
                    'relevance_score': 0.9,
                    'key_terms': ['supervised learning', 'machine learning']
                }
            ],
            'summary': 'Machine Learning Overview contains relevant information about supervised learning'
        }
    ]
    
    test_metadata = {
        'documents': ['ml_textbook.pdf', 'neural_nets.pdf'],
        'persona': 'I am a data scientist with expertise in machine learning',
        'job': 'Find information about neural networks and machine learning algorithms',
        'total_sections': 10,
        'model_info': {
            'model_name': 'all-MiniLM-L6-v2',
            'model_loaded': True,
            'dependencies_available': True
        }
    }
    
    test_persona_context = {
        'persona': {
            'role': 'data_scientist',
            'domain': 'technical',
            'expertise_level': 'expert',
            'expertise_areas': ['machine learning', 'neural networks']
        },
        'job': {
            'job_type': 'analysis',
            'main_goals': ['find', 'analyze'],
            'deliverable_type': 'information',
            'urgency': 'normal'
        }
    }
    
    # Test output generation
    output = generator.generate_json_output(
        test_ranked_sections,
        test_subsection_analysis,
        test_metadata,
        test_persona_context
    )
    
    print(f"✓ Generated output structure:")
    print(f"  Metadata fields: {len(output['metadata'])}")
    print(f"  Extracted sections: {len(output['extracted_sections'])}")
    print(f"  Subsection analysis: {len(output['subsection_analysis'])}")
    
    # Test validation
    try:
        generator._validate_output(output)
        print("✓ Output validation passed")
    except Exception as e:
        print(f"✗ Output validation failed: {e}")
    
    # Test metadata generation
    metadata = generator._generate_metadata(test_ranked_sections, test_metadata, test_persona_context)
    print(f"✓ Metadata generated with persona role: {metadata['persona_analysis']['extracted_role']}")
    print(f"✓ Job type identified: {metadata['job_analysis']['primary_job_type']}")
    
    # Test section formatting
    formatted_sections = generator._format_extracted_sections(test_ranked_sections)
    print(f"✓ Formatted {len(formatted_sections)} sections")
    if formatted_sections:
        top_section = formatted_sections[0]
        print(f"  Top section: {top_section['section_title']} (score: {top_section['relevance_score']})")
        print(f"  Word count: {top_section['content']['word_count']}")
    
    # Test score distribution analysis
    score_dist = generator._analyze_score_distribution(test_ranked_sections)
    print(f"✓ Score distribution: min={score_dist['min_score']:.3f}, max={score_dist['max_score']:.3f}")
    
    # Test document coverage analysis
    doc_coverage = generator._analyze_document_coverage(test_ranked_sections)
    print(f"✓ Document coverage: {doc_coverage['documents_represented']} documents represented")
    
    print("Output Generator test completed successfully!")
    return True

if __name__ == "__main__":
    test_output_generator()