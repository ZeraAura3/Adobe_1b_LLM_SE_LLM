# Round 1B: Persona-Driven Document Intelligence

## Project Overview
This solution extracts and prioritizes relevant sections from PDF document collections based on a specific persona and their job-to-be-done requirements.

## Features
- ✅ **CPU-Only Processing**: No GPU dependencies, optimized for CPU execution
- ✅ **Fast Performance**: Processes 3-5 documents in <60 seconds
- ✅ **Offline Operation**: No internet access required during execution
- ✅ **Generic Solution**: Works across diverse domains, personas, and tasks
- ✅ **Lightweight Models**: Total model size <1GB
- ✅ **Structured Output**: JSON format with relevance rankings

## Architecture

### Core Components
1. **DocumentProcessor**: PDF text extraction and section detection
2. **PersonaJobAnalyzer**: Persona and job requirement analysis
3. **RelevanceScorer**: TF-IDF based section relevance scoring
4. **SubSectionAnalyzer**: Granular sub-section extraction and ranking

### Processing Pipeline
```
PDFs → Document Processing → Persona/Job Analysis → Relevance Scoring → Sub-section Analysis → JSON Output
```

## Installation & Usage

### Docker Build
```bash
docker build --platform linux/amd64 -t persona-doc-intelligence:latest .
```

### Docker Run
```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  persona-doc-intelligence:latest
```

### Input Requirements
- **PDF Files**: Place all PDF documents in `/app/input/` directory
- **Persona Description**: Create `/app/input/persona.txt` with role description
- **Job Description**: Create `/app/input/job.txt` with task requirements

### Output
- **Result File**: `/app/output/challenge1b_output.json`
- **Format**: Structured JSON with metadata, extracted sections, and sub-section analysis

## Example Input

### persona.txt
```
I am a PhD researcher in computational biology with 5 years of research experience. 
I specialize in machine learning applications to drug discovery and have strong 
analytical skills in methodology evaluation.
```

### job.txt
```
Prepare a comprehensive literature review focusing on methodologies, datasets, 
and performance benchmarks for graph neural networks in drug discovery applications.
```

## Technical Specifications

### Dependencies
- **PDF Processing**: pdfplumber, PyPDF2
- **NLP**: spaCy (en_core_web_sm), NLTK
- **ML**: scikit-learn, numpy, pandas
- **Python**: 3.9+ with standard libraries

### Performance
- **Processing Time**: <60 seconds for 3-5 documents
- **Memory Usage**: <2GB RAM
- **Model Size**: ~200MB (spaCy model + libraries)
- **CPU Architecture**: AMD64 (x86_64)

### Constraints Compliance
- ✅ CPU-only execution
- ✅ Model size ≤ 1GB  
- ✅ Processing time ≤ 60 seconds
- ✅ No internet access required
- ✅ Generic across domains

## File Structure
```
.
├── Dockerfile                 # Container configuration
├── requirements.txt          # Python dependencies
├── main.py                  # Main execution script
├── document_processor.py    # PDF processing module
├── persona_analyzer.py      # Persona/job analysis module
├── relevance_scorer.py      # Section relevance scoring
├── subsection_analyzer.py   # Sub-section extraction
├── approach_explanation.md  # Methodology documentation
└── README.md               # This file
```

## Output Format

The system generates a JSON file with the following structure:

```json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "PhD researcher in computational biology...",
    "job_to_be_done": "Prepare a comprehensive literature review...",
    "processing_timestamp": "2025-01-28T10:30:00",
    "total_processing_time_seconds": 45.2
  },
  "extracted_sections": [
    {
      "document": "doc1.pdf",
      "page_number": 3,
      "section_title": "Methodology",
      "importance_rank": 1,
      "relevance_score": 0.8534,
      "relevance_factors": ["Title matches: methodology", "Domain relevance"]
    }
  ],
  "subsection_analysis": [
    {
      "document": "doc1.pdf",
      "parent_section": "Methodology",
      "refined_text": "The proposed methodology combines...",
      "page_number": 3,
      "importance_rank": 1,
      "relevance_score": 0.7823,
      "word_count": 156,
      "extraction_method": "paragraph_split"
    }
  ]
}
```

## Development Notes

### Fallback Mechanisms
The system includes fallback mechanisms for environments where certain libraries might not be available:
- Keyword-based scoring when sklearn is unavailable
- Simple sentence tokenization when NLTK is unavailable
- Basic text processing when spaCy models are missing

### Error Handling
- Robust PDF processing with multiple extraction methods
- Graceful degradation when models are unavailable
- Comprehensive logging for debugging

### Optimization
- Memory-efficient processing for large document collections
- CPU-optimized algorithms with minimal computational overhead
- Streaming processing to handle memory constraints

## License
This project is developed for the Adobe Challenge Round 1B.

---

**Ready for submission!** This solution meets all requirements for the Persona-Driven Document Intelligence challenge.
