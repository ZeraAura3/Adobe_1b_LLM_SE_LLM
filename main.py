#!/usr/bin/env python3
"""
Round 1B: Persona-Driven Document Intelligence
Main execution script for the Adobe Challenge

This script processes a collection of PDF documents based on a given persona
and job-to-be-done, extracting and ranking the most relevant sections.

Usage:
    python main.py
    
The script expects:
    - Input PDFs in /app/input/ directory
    - persona.txt and job.txt files in /app/input/
    - Outputs results to /app/output/challenge1b_output.json
"""

import json
import os
import time
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# PDF Processing
import PyPDF2
import pdfplumber
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

# NLP and Text Processing (lightweight, CPU-friendly)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import spacy

# Mathematical and data processing
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict

# Import our custom modules
from document_processor import DocumentProcessor
from persona_analyzer import PersonaJobAnalyzer
from relevance_scorer import RelevanceScorer
from subsection_analyzer import SubSectionAnalyzer


class PersonaDrivenDocumentIntelligence:
    """Main orchestrator class for the complete pipeline."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.persona_analyzer = PersonaJobAnalyzer()
        self.relevance_scorer = RelevanceScorer()
        self.subsection_analyzer = SubSectionAnalyzer()
        
        self.processing_start_time = None
        self.results = None
        
        # Initialize NLTK data
        self._download_nltk_data()
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️  SpaCy model not found. Using basic NLP processing.")
            self.nlp = None
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        nltk_data = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
        for data in nltk_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                try:
                    nltk.download(data, quiet=True)
                except:
                    print(f"⚠️  Could not download NLTK data: {data}")
    
    def process(self, pdf_paths: List[str], persona_description: str, 
                job_description: str, output_path: str = None) -> Dict[str, Any]:
        """Main processing pipeline."""
        
        self.processing_start_time = time.time()
        print("🚀 Starting Persona-Driven Document Intelligence Pipeline")
        
        # Step 1: Process documents
        print("📚 Step 1: Processing documents...")
        documents = self.document_processor.process_documents(pdf_paths)
        
        if not documents:
            raise ValueError("❌ No documents were successfully processed")
        
        # Step 2: Analyze persona and job
        print("🧑‍💼 Step 2: Analyzing persona and job requirements...")
        persona_analysis = self.persona_analyzer.analyze_persona(persona_description)
        job_analysis = self.persona_analyzer.analyze_job(job_description)
        requirements = self.persona_analyzer.combine_requirements(persona_analysis, job_analysis)
        
        # Step 3: Score relevance
        print("📊 Step 3: Scoring section relevance...")
        self.relevance_scorer.prepare_scoring(documents, requirements)
        scored_sections = self.relevance_scorer.calculate_relevance_scores(requirements)
        top_sections = self.relevance_scorer.extract_top_sections(scored_sections, top_n=10)
        
        # Step 4: Analyze sub-sections
        print("🔍 Step 4: Analyzing sub-sections...")
        sub_sections = self.subsection_analyzer.analyze_subsections(
            documents, top_sections, requirements, max_subsections=20
        )
        
        # Step 5: Generate output
        print("📄 Step 5: Generating output...")
        output = self._generate_output(
            pdf_paths, persona_description, job_description,
            top_sections, sub_sections, persona_analysis, job_analysis
        )
        
        total_time = time.time() - self.processing_start_time
        print(f"🎉 Pipeline completed in {total_time:.2f}s")
        
        if total_time > 60:
            print("⚠️  Warning: Processing time exceeded 60s constraint!")
        
        # Save output if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"💾 Output saved to: {output_path}")
        
        self.results = output
        return output
    
    def _generate_output(self, pdf_paths: List[str], persona: str, job: str,
                        top_sections: List[Dict], sub_sections: List[Dict],
                        persona_analysis: Dict, job_analysis: Dict) -> Dict[str, Any]:
        """Generate the required JSON output format."""
        
        # Metadata
        metadata = {
            "input_documents": [Path(path).name for path in pdf_paths],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat(),
            "total_processing_time_seconds": round(time.time() - self.processing_start_time, 2),
            "persona_analysis": {
                "role": persona_analysis.get('role', ''),
                "domain": persona_analysis.get('domain', ''),
                "experience_level": persona_analysis.get('experience_level', '')
            },
            "job_analysis": {
                "task_type": job_analysis.get('task_type', ''),
                "key_concepts": job_analysis.get('key_concepts', [])[:5]
            }
        }
        
        # Extracted sections
        extracted_sections = []
        for section in top_sections:
            extracted_sections.append({
                "document": section['document'],
                "page_number": section['page_number'],
                "section_title": section['section_title'],
                "importance_rank": section['importance_rank'],
                "relevance_score": section['relevance_score'],
                "relevance_factors": section.get('relevance_factors', [])
            })
        
        # Sub-section analysis
        subsection_analysis = []
        for subsection in sub_sections:
            subsection_analysis.append({
                "document": subsection['document'],
                "parent_section": subsection['parent_section'],
                "refined_text": subsection['refined_text'],
                "page_number": subsection['page_number'],
                "importance_rank": subsection['importance_rank'],
                "relevance_score": subsection['relevance_score'],
                "word_count": subsection['word_count'],
                "extraction_method": subsection['extraction_method']
            })
        
        return {
            "metadata": metadata,
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }


def main():
    """Main execution function."""
    
    # Define input/output paths
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files found in /app/input directory")
        print("📋 Please ensure PDF files are placed in the input directory")
        # Don't return, continue with empty list for testing
        pdf_files = []
    
    if pdf_files:
        print(f"📁 Found {len(pdf_files)} PDF files: {[f.name for f in pdf_files]}")
    else:
        print("⚠️  No PDF files found, running with minimal configuration for testing")
    
    # Read persona and job descriptions
    try:
        persona_file = input_dir / "persona.txt"
        job_file = input_dir / "job.txt"
        
        if persona_file.exists():
            persona = persona_file.read_text(encoding='utf-8').strip()
        else:
            # Default persona if file not found
            persona = "I am a researcher with experience in document analysis and information extraction."
            print("⚠️  persona.txt not found, using default persona")
        
        if job_file.exists():
            job = job_file.read_text(encoding='utf-8').strip()
        else:
            # Default job if file not found
            job = "Extract and analyze the most relevant sections from the provided documents."
            print("⚠️  job.txt not found, using default job description")
        
    except Exception as e:
        print(f"❌ Error reading persona/job files: {e}")
        return
    
    # Initialize and run the system
    try:
        intelligence_system = PersonaDrivenDocumentIntelligence()
        
        if pdf_files:  # Only process if PDFs are found
            results = intelligence_system.process(
                pdf_paths=[str(f) for f in pdf_files],
                persona_description=persona,
                job_description=job,
                output_path=str(output_dir / "challenge1b_output.json")
            )
            
            print("✅ Processing completed successfully!")
        else:
            # Create minimal output for testing without PDFs
            minimal_output = {
                "metadata": {
                    "input_documents": [],
                    "persona": persona,
                    "job_to_be_done": job,
                    "processing_timestamp": "2025-01-28T00:00:00",
                    "total_processing_time_seconds": 0.0,
                    "note": "No PDF files provided for processing"
                },
                "extracted_sections": [],
                "subsection_analysis": []
            }
            
            import json
            with open(output_dir / "challenge1b_output.json", 'w') as f:
                json.dump(minimal_output, f, indent=2)
            
            print("⚠️  No PDFs processed, created minimal output file")
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
