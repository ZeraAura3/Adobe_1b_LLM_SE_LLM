#!/usr/bin/env python3
"""
Round 1B: Main Pipeline for Persona-Driven Document Intelligence
Integrates all components for end-to-end processing
"""

import os
import sys
import json
import time
import logging
import shutil
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import project modules
from pdf_processor import PDFProcessor
from persona_analyzer import PersonaAnalyzer
from relevance_scorer import RelevanceScorer
from output_generator import OutputGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('processing.log') if True else logging.NullHandler()  # Local log file
    ]
)
logger = logging.getLogger(__name__)

class Round1BPipeline:
    def __init__(self):
        """Initialize all pipeline components"""
        logger.info("Initializing Round 1B Pipeline...")
        
        self.pdf_processor = PDFProcessor()
        self.persona_analyzer = PersonaAnalyzer()
        self.scorer = RelevanceScorer()
        self.output_generator = OutputGenerator()
        
        # Configuration - Use local folders in Round1b directory
        project_root = Path(__file__).parent.parent  # Go up to Round1b folder
        self.config = {
            'pdf_dir': str(project_root / 'pdf'),      # PDF files folder
            'input_dir': str(project_root / 'input'),  # Persona/job text files
            'output_dir': str(project_root / 'output'), # Results folder
            'top_k_sections': 10,
            'enable_debug': True
        }
        
        logger.info("Pipeline initialized successfully")

    def collect_user_inputs(self) -> Dict:
        """Interactively collect user inputs and save them to appropriate folders"""
        print("\n" + "="*60)
        print("🎯 PERSONA-DRIVEN DOCUMENT INTELLIGENCE")
        print("="*60)
        print("Please provide the following information:")
        print("(Press Enter to skip and use existing files if available)")
        print()
        
        inputs = {
            'pdf_provided': False, 
            'persona_provided': False, 
            'job_provided': False,
            'process_single_pdf': False,
            'new_pdf_filename': None
        }
        
        # Ask for PDF file
        print("1️⃣ PDF Document:")
        pdf_path = input("   Enter path to PDF file (or press Enter to skip): ").strip()
        
        if pdf_path and os.path.exists(pdf_path) and pdf_path.lower().endswith('.pdf'):
            try:
                # Copy PDF to pdf folder
                pdf_filename = Path(pdf_path).name
                destination = Path(self.config['pdf_dir']) / pdf_filename
                os.makedirs(self.config['pdf_dir'], exist_ok=True)
                shutil.copy2(pdf_path, destination)
                print(f"   ✅ PDF copied to: {destination}")
                inputs['pdf_provided'] = True
                inputs['new_pdf_filename'] = pdf_filename
                
                # Ask about processing scope
                print("\n   📋 Processing Scope:")
                existing_pdfs = list(Path(self.config['pdf_dir']).glob("*.pdf"))
                if len(existing_pdfs) > 1:
                    print(f"   Found {len(existing_pdfs)} PDF files in total (including your new one)")
                    print("   Options:")
                    print(f"   1. Process only your new PDF: {pdf_filename}")
                    print(f"   2. Process all {len(existing_pdfs)} PDFs in the folder")
                    
                    while True:
                        choice = input("   Choose option (1 or 2): ").strip()
                        if choice == "1":
                            inputs['process_single_pdf'] = True
                            print(f"   ✅ Will process only: {pdf_filename}")
                            break
                        elif choice == "2":
                            inputs['process_single_pdf'] = False
                            print(f"   ✅ Will process all {len(existing_pdfs)} PDFs")
                            break
                        else:
                            print("   ❌ Please enter 1 or 2")
                else:
                    print("   ✅ Will process your PDF (only one in folder)")
                    inputs['process_single_pdf'] = True
                    
            except Exception as e:
                print(f"   ❌ Error copying PDF: {e}")
        elif pdf_path:
            print(f"   ❌ Invalid PDF path: {pdf_path}")
        else:
            # No new PDF provided, check if existing PDFs are available
            existing_pdfs = list(Path(self.config['pdf_dir']).glob("*.pdf")) if os.path.exists(self.config['pdf_dir']) else []
            if existing_pdfs:
                print(f"   📋 Found {len(existing_pdfs)} existing PDF file(s) in the folder:")
                for i, pdf in enumerate(existing_pdfs[:5], 1):  # Show first 5
                    print(f"      {i}. {pdf.name}")
                if len(existing_pdfs) > 5:
                    print(f"      ... and {len(existing_pdfs) - 5} more")
                
                use_existing = input("   Do you want to process these existing PDFs? (y/n): ").strip().lower()
                if use_existing in ['y', 'yes']:
                    inputs['process_single_pdf'] = False  # Process all existing
                    print(f"   ✅ Will process all {len(existing_pdfs)} existing PDFs")
                else:
                    print("   ⏭️  Skipped existing PDFs")
        
        # Ask for persona
        print("\n2️⃣ Your Persona:")
        print("   Describe yourself (role, expertise, experience, interests)")
        persona = input("   Enter persona description (or press Enter to skip): ").strip()
        
        if persona:
            try:
                os.makedirs(self.config['input_dir'], exist_ok=True)
                persona_file = Path(self.config['input_dir']) / 'persona.txt'
                persona_file.write_text(persona, encoding='utf-8')
                print(f"   ✅ Persona saved to: {persona_file}")
                inputs['persona_provided'] = True
            except Exception as e:
                print(f"   ❌ Error saving persona: {e}")
        
        # Ask for job
        print("\n3️⃣ Your Task/Job:")
        print("   Describe what you're looking for or need to accomplish")
        job = input("   Enter job description (or press Enter to skip): ").strip()
        
        if job:
            try:
                os.makedirs(self.config['input_dir'], exist_ok=True)
                job_file = Path(self.config['input_dir']) / 'job.txt'
                job_file.write_text(job, encoding='utf-8')
                print(f"   ✅ Job description saved to: {job_file}")
                inputs['job_provided'] = True
            except Exception as e:
                print(f"   ❌ Error saving job: {e}")
        
        print("\n" + "-"*60)
        return inputs

    def load_inputs(self) -> Dict:
        """Load persona and job inputs from files in input folder"""
        try:
            persona, job = self._load_from_files()
            
            if not persona and not job:
                logger.error("No data available - no persona or job description found")
                return {'persona': '', 'job': '', 'valid': False}
            
            if not persona:
                logger.warning("No persona description found")
            if not job:
                logger.warning("No job description found")
            
            logger.info(f"Loaded inputs - Persona: {persona[:50] if persona else 'None'}...")
            logger.info(f"Job: {job[:50] if job else 'None'}...")
            
            return {'persona': persona, 'job': job, 'valid': bool(persona or job)}
            
        except Exception as e:
            logger.error(f"Error loading inputs: {e}")
            return {'persona': '', 'job': '', 'valid': False}

    def _load_from_files(self) -> tuple:
        """Load persona and job from input folder files"""
        persona = ""
        job = ""
        
        try:
            # Check for persona.txt and job.txt in input directory
            input_path = Path(self.config['input_dir'])
            
            persona_file = input_path / 'persona.txt'
            job_file = input_path / 'job.txt'
            
            if persona_file.exists():
                persona = persona_file.read_text(encoding='utf-8').strip()
                logger.info(f"Loaded persona from {persona_file}")
            
            if job_file.exists():
                job = job_file.read_text(encoding='utf-8').strip()
                logger.info(f"Loaded job from {job_file}")
                
        except Exception as e:
            logger.warning(f"Could not load from files: {e}")
        
        return persona, job

    def process_documents(self, user_inputs: Dict = None) -> List[Dict]:
        """Process PDF documents based on user preferences"""
        logger.info("Starting document processing...")
        start_time = time.time()
        
        try:
            pdf_dir = self.config['pdf_dir']
            if not os.path.exists(pdf_dir):
                logger.error(f"PDF directory does not exist: {pdf_dir}")
                return []
            
            # Check if there are any PDF files
            all_pdf_files = list(Path(pdf_dir).glob("*.pdf"))
            if not all_pdf_files:
                logger.error("NO PDF is available in the pdf folder")
                logger.info(f"Please add PDF files to: {pdf_dir}")
                return []
            
            # Determine which PDFs to process
            pdf_files_to_process = []
            
            if user_inputs and user_inputs.get('process_single_pdf') and user_inputs.get('new_pdf_filename'):
                # Process only the newly provided PDF
                target_file = Path(pdf_dir) / user_inputs['new_pdf_filename']
                if target_file.exists():
                    pdf_files_to_process = [target_file]
                    logger.info(f"Processing single PDF: {user_inputs['new_pdf_filename']}")
                else:
                    logger.warning(f"Target PDF not found: {target_file}")
                    pdf_files_to_process = all_pdf_files
            else:
                # Process all PDFs in the folder
                pdf_files_to_process = all_pdf_files
                logger.info(f"Processing all PDFs in folder")
            
            if not pdf_files_to_process:
                logger.error("No valid PDF files to process")
                return []
            
            logger.info(f"Found {len(pdf_files_to_process)} PDF file(s) to process...")
            
            # Process selected PDFs
            all_sections = []
            for pdf_file in pdf_files_to_process:
                logger.info(f"Processing: {pdf_file}")
                try:
                    # Process individual PDF
                    result = self.pdf_processor.extract_text_with_structure(str(pdf_file))
                    
                    # Add document info to each section
                    for section in result['sections']:
                        section['document'] = result['document']
                        section['pdf_path'] = str(pdf_file)
                    
                    all_sections.extend(result['sections'])
                    
                except Exception as e:
                    logger.error(f"Error processing {pdf_file}: {e}")
                    continue
            
            processing_time = time.time() - start_time
            logger.info(f"Document processing completed in {processing_time:.2f} seconds")
            logger.info(f"Extracted {len(all_sections)} sections from {len(pdf_files_to_process)} documents")
            
            return all_sections
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return []

    def analyze_and_rank(self, all_sections: List[Dict], inputs: Dict) -> tuple:
        """Analyze persona/job and rank sections"""
        logger.info("Starting persona analysis and section ranking...")
        start_time = time.time()
        
        try:
            # Analyze persona and job
            persona_context = self.persona_analyzer.create_context(
                inputs['persona'], 
                inputs['job']
            )
            
            logger.info(f"Identified domain: {persona_context.get('persona', {}).get('domain', 'unknown')}")
            logger.info(f"Identified role: {persona_context.get('persona', {}).get('role', 'unknown')}")
            logger.info(f"Job type: {persona_context.get('job', {}).get('job_type', 'unknown')}")
            
            # Rank sections by relevance
            ranked_sections = self.scorer.rank_sections(
                all_sections, 
                persona_context, 
                top_k=self.config['top_k_sections']
            )
            
            # Analyze subsections
            subsection_analysis = self.scorer.analyze_subsections(ranked_sections, persona_context)
            
            analysis_time = time.time() - start_time
            logger.info(f"Analysis and ranking completed in {analysis_time:.2f} seconds")
            
            return ranked_sections, subsection_analysis, persona_context
            
        except Exception as e:
            logger.error(f"Error in analysis and ranking: {e}")
            return [], [], {}

    def generate_output(self, ranked_sections: List[Dict], subsection_analysis: List[Dict], 
                       persona_context: Dict, inputs: Dict, all_sections: List[Dict]) -> Dict:
        """Generate final output"""
        logger.info("Generating output...")
        
        try:
            # Prepare metadata
            metadata = {
                'documents': list(set([s.get('document', '') for s in all_sections])),
                'persona': inputs['persona'],
                'job': inputs['job'],
                'total_sections': len(all_sections),
                'model_info': self.scorer.get_model_info()
            }
            
            # Generate JSON output
            output = self.output_generator.generate_json_output(
                ranked_sections,
                subsection_analysis,
                metadata,
                persona_context
            )
            
            logger.info("Output generation completed")
            return output
            
        except Exception as e:
            logger.error(f"Error generating output: {e}")
            return {}

    def save_results(self, output: Dict, ranked_sections: List[Dict]) -> bool:
        """Save all results to output directory"""
        logger.info("Saving results...")
        
        try:
            output_dir = self.config['output_dir']
            os.makedirs(output_dir, exist_ok=True)
            
            # Save main result
            main_output_path = os.path.join(output_dir, 'result.json')
            success = self.output_generator.save_output(output, main_output_path)
            
            if not success:
                logger.error("Failed to save main output")
                return False
            
            # Save additional files if debug enabled
            if self.config['enable_debug']:
                self.output_generator.save_debug_info(ranked_sections, output_dir)
                self.output_generator.generate_summary_report(output, output_dir)
            
            # Always save compiled relevant information as text file
            self.save_relevant_info_text(output, ranked_sections, output_dir)
            
            logger.info(f"Results saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False

    def save_relevant_info_text(self, output: Dict, ranked_sections: List[Dict], output_dir: str) -> bool:
        """Save compiled relevant information as a readable text file"""
        try:
            text_file_path = os.path.join(output_dir, 'relevant_information.txt')
            
            with open(text_file_path, 'w', encoding='utf-8') as f:
                # Header
                f.write("="*80 + "\n")
                f.write("RELEVANT INFORMATION EXTRACTION REPORT\n")
                f.write("="*80 + "\n\n")
                
                # Metadata
                metadata = output.get('metadata', {})
                f.write("📊 ANALYSIS SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"📅 Generated: {metadata.get('processing_timestamp', 'Unknown')}\n")
                f.write(f"📄 Documents: {', '.join(metadata.get('input_documents', ['Unknown']))}\n")
                f.write(f"📝 Total Sections Analyzed: {metadata.get('total_sections_analyzed', 0)}\n")
                f.write(f"🎯 Top Sections Selected: {metadata.get('top_sections_selected', 0)}\n\n")
                
                # Persona & Job Analysis
                f.write("👤 YOUR PROFILE\n")
                f.write("-" * 40 + "\n")
                f.write(f"Persona: {metadata.get('persona', 'Not provided')}\n")
                f.write(f"Task: {metadata.get('job_to_be_done', 'Not provided')}\n\n")
                
                persona_analysis = metadata.get('persona_analysis', {})
                f.write(f"🔍 Analysis Results:\n")
                f.write(f"   • Role: {persona_analysis.get('extracted_role', 'Unknown')}\n")
                f.write(f"   • Domain: {persona_analysis.get('identified_domain', 'Unknown')}\n")
                f.write(f"   • Expertise: {persona_analysis.get('expertise_level', 'Unknown')}\n")
                f.write(f"   • Search Terms: {persona_analysis.get('key_search_terms', 0)} generated\n\n")
                
                # Model Information
                model_info = metadata.get('model_info', {})
                f.write("🤖 PROCESSING DETAILS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Model: {model_info.get('model_name', 'Unknown')}\n")
                f.write(f"Dependencies Available: {model_info.get('dependencies_available', False)}\n")
                weights = model_info.get('weights', {})
                f.write(f"Scoring Weights: Semantic {weights.get('semantic', 0):.0%}, ")
                f.write(f"Lexical {weights.get('lexical', 0):.0%}, ")
                f.write(f"Title {weights.get('title_match', 0):.0%}\n\n")
                
                # Top Relevant Sections
                f.write("🎯 TOP RELEVANT SECTIONS\n")
                f.write("="*80 + "\n\n")
                
                extracted_sections = output.get('extracted_sections', [])
                
                for i, section in enumerate(extracted_sections, 1):
                    f.write(f"SECTION {i}\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"📊 Relevance Score: {section.get('relevance_score', 0):.4f}\n")
                    f.write(f"📄 Document: {section.get('document_name', 'Unknown')}\n")
                    f.write(f"📑 Title: {section.get('section_title', 'No title')}\n")
                    f.write(f"📃 Page: {section.get('page_number', 'Unknown')}\n")
                    f.write(f"🔢 Heading Level: {section.get('heading_level', 'Unknown')}\n")
                    
                    content = section.get('content', {})
                    f.write(f"📝 Word Count: {content.get('word_count', 0)}\n")
                    f.write(f"🔤 Character Count: {content.get('character_count', 0)}\n")
                    
                    # Relevance indicators
                    indicators = section.get('relevance_indicators', {})
                    f.write(f"🎯 Contains High Priority Terms: {'Yes' if indicators.get('contains_high_priority_terms') else 'No'}\n")
                    f.write(f"🎯 Contains Medium Priority Terms: {'Yes' if indicators.get('contains_medium_priority_terms') else 'No'}\n")
                    f.write(f"🎯 Title Relevance: {'Yes' if indicators.get('title_relevance') else 'No'}\n")
                    
                    # Content text
                    text_content = content.get('text', '').strip()
                    if text_content:
                        f.write(f"\n📖 CONTENT:\n")
                        f.write("-" * 30 + "\n")
                        # Limit content to first 500 characters for readability
                        if len(text_content) > 500:
                            f.write(f"{text_content[:500]}...\n")
                            f.write(f"[Content truncated - full text has {len(text_content)} characters]\n")
                        else:
                            f.write(f"{text_content}\n")
                    else:
                        f.write(f"\n📖 CONTENT: [No text content available]\n")
                    
                    f.write("\n" + "="*80 + "\n\n")
                
                # Summary Statistics
                f.write("📈 EXTRACTION STATISTICS\n")
                f.write("-" * 40 + "\n")
                
                # Calculate statistics
                total_words = sum(section.get('content', {}).get('word_count', 0) for section in extracted_sections)
                total_chars = sum(section.get('content', {}).get('character_count', 0) for section in extracted_sections)
                sections_with_content = sum(1 for section in extracted_sections if section.get('content', {}).get('text', '').strip())
                avg_relevance = sum(section.get('relevance_score', 0) for section in extracted_sections) / len(extracted_sections) if extracted_sections else 0
                
                f.write(f"📊 Total Extracted Words: {total_words:,}\n")
                f.write(f"📊 Total Extracted Characters: {total_chars:,}\n")
                f.write(f"📊 Sections with Content: {sections_with_content}/{len(extracted_sections)}\n")
                f.write(f"📊 Average Relevance Score: {avg_relevance:.4f}\n")
                
                # Page distribution
                pages = [section.get('page_number', 0) for section in extracted_sections if section.get('page_number')]
                if pages:
                    f.write(f"📊 Page Range: {min(pages)} - {max(pages)}\n")
                    f.write(f"📊 Unique Pages: {len(set(pages))}\n")
                
                f.write(f"\n📁 Full detailed results available in: result.json\n")
                f.write(f"📁 Debug information available in: debug_info.json\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("Report generated by Persona-Driven Document Intelligence System\n")
                f.write("="*80 + "\n")
            
            logger.info(f"Relevant information saved to: {text_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving relevant information text: {e}")
            return False

    def run(self) -> bool:
        """Run the complete pipeline"""
        logger.info("="*60)
        logger.info("Starting Round 1B: Persona-Driven Document Intelligence")
        logger.info("="*60)
        
        total_start_time = time.time()
        
        try:
            # Step 0: Collect user inputs interactively
            user_inputs = self.collect_user_inputs()
            
            # Step 1: Load inputs from files
            inputs = self.load_inputs()
            if not inputs.get('valid', False):
                logger.error("No data available - please provide persona and/or job description")
                print("\n❌ No data available!")
                print("   Please provide at least a persona description or job description.")
                return False
            
            # Step 2: Process documents
            all_sections = self.process_documents(user_inputs)
            if not all_sections:
                logger.error("No sections extracted from documents")
                logger.info("Please ensure PDF files are placed in the 'pdf' folder and try again")
                print("\n❌ No PDF documents found!")
                print("   Please add PDF files to the 'pdf' folder.")
                return False
            
            # Step 3: Analyze and rank
            ranked_sections, subsection_analysis, persona_context = self.analyze_and_rank(all_sections, inputs)
            if not ranked_sections:
                logger.error("No sections ranked - check persona/job relevance")
                return False
            
            # Step 4: Generate output
            output = self.generate_output(ranked_sections, subsection_analysis, persona_context, inputs, all_sections)
            if not output:
                logger.error("Failed to generate output")
                return False
            
            # Step 5: Save results
            success = self.save_results(output, ranked_sections)
            if not success:
                logger.error("Failed to save results")
                return False
            
            # Report completion
            total_time = time.time() - total_start_time
            logger.info("="*60)
            logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
            logger.info(f"Top section relevance score: {ranked_sections[0]['score']:.3f}")
            logger.info(f"Results saved with {len(ranked_sections)} sections")
            logger.info("="*60)
            
            print(f"\n🎉 SUCCESS! Analysis completed in {total_time:.1f} seconds")
            print(f"📊 Found {len(ranked_sections)} relevant sections")
            print(f"📁 Results saved to: {self.config['output_dir']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return False

def main():
    """Main entry point"""
    try:
        pipeline = Round1BPipeline()
        success = pipeline.run()
        
        if success:
            logger.info("Round 1B pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("Round 1B pipeline failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

def test_main_pipeline():
    """Test main pipeline functionality"""
    print("Testing Main Pipeline...")
    
    # Test pipeline initialization
    pipeline = Round1BPipeline()
    print("✓ Pipeline initialized successfully")
    persona = input("Enter persona (or leave empty for default): ") or "A professional seeking relevant information"
    job = input("Enter job (or leave empty for default): ") or "Find the most relevant sections for my work"    
    # Test input loading with sample data
    sample_inputs = {
        'persona': persona,
        'job': job
    }
    
    print(f"✓ Sample inputs prepared:")
    print(f"  Persona: {sample_inputs['persona'][:50]}...")
    print(f"  Job: {sample_inputs['job'][:50]}...")
    
    # Test components individually
    print("\n--- Testing Individual Components ---")
    
    # Test persona analyzer
    persona_context = pipeline.persona_analyzer.create_context(
        sample_inputs['persona'], 
        sample_inputs['job']
    )
    print(f"✓ Persona analysis: domain={persona_context['persona']['domain']}, role={persona_context['persona']['role']}")
    
    # Create sample sections for testing
    sample_sections = [
        {
            'title': 'Transformer Architecture',
            'text': 'This section discusses the transformer architecture, attention mechanisms, and self-attention in neural networks.',
            'page': 1,
            'level': 1,
            'document': 'transformer_paper.pdf',
            'pdf_path': '/sample/transformer_paper.pdf'
        },
        {
            'title': 'Experimental Setup',
            'text': 'We describe the experimental setup used to evaluate our proposed method on various benchmarks.',
            'page': 3,
            'level': 2,
            'document': 'transformer_paper.pdf',
            'pdf_path': '/sample/transformer_paper.pdf'
        }
    ]
    
    # Test relevance scoring
    ranked_sections = pipeline.scorer.rank_sections(sample_sections, persona_context, top_k=5)
    print(f"✓ Section ranking: {len(ranked_sections)} sections ranked")
    if ranked_sections:
        print(f"  Top section: {ranked_sections[0]['title']} (score: {ranked_sections[0]['score']:.3f})")
    
    # Test subsection analysis
    subsection_analysis = pipeline.scorer.analyze_subsections(ranked_sections, persona_context)
    print(f"✓ Subsection analysis: {len(subsection_analysis)} analyses generated")
    
    # Test output generation
    metadata = {
        'documents': ['transformer_paper.pdf'],
        'persona': sample_inputs['persona'],
        'job': sample_inputs['job'],
        'total_sections': len(sample_sections),
        'model_info': pipeline.scorer.get_model_info()
    }
    
    output = pipeline.output_generator.generate_json_output(
        ranked_sections,
        subsection_analysis,
        metadata,
        persona_context
    )
    print(f"✓ Output generation: {len(output['extracted_sections'])} sections in output")
    
    # Test configuration
    print(f"\n--- Pipeline Configuration ---")
    for key, value in pipeline.config.items():
        print(f"  {key}: {value}")
    
    print("\nMain Pipeline test completed successfully!")
    return True

if __name__ == "__main__":
    # Check if we want to run tests or the main pipeline
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_main_pipeline()
    else:
        main()