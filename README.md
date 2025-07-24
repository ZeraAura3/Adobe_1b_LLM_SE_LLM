# Round 1B: Persona-Driven Document Intelligence

## Overview

This solution implements a sophisticated persona-driven document intelligence system that extracts and ranks document sections based on user personas and specific job requirements. The system combines structural document analysis with semantic understanding to deliver highly relevant content.

## Architecture

The solution consists of four main components:

1. **PDF Processor** (`pdf_processor.py`) - Extracts text with structural mapping
2. **Persona Analyzer** (`persona_analyzer.py`) - Analyzes user personas and job requirements
3. **Relevance Scorer** (`relevance_scorer.py`) - Ranks sections using hybrid scoring
4. **Output Generator** (`output_generator.py`) - Generates structured JSON output

## Key Features

- **Hybrid Relevance Scoring**: Combines semantic similarity (sentence transformers) with lexical matching
- **Persona-Aware Analysis**: Extracts role, domain, expertise level, and terminology from user descriptions
- **Job-Driven Ranking**: Prioritizes content based on specific task requirements
- **Structural Understanding**: Maintains document hierarchy and section relationships
- **Robust Processing**: Handles multiple document formats and edge cases

## Model Information

- **Sentence Transformer**: `all-MiniLM-L6-v2` (~80MB)
- **Semantic Similarity**: Cosine similarity on document embeddings
- **Lexical Matching**: Fuzzy string matching with RapidFuzz
- **Scoring Weights**: 60% semantic, 25% lexical, 15% title matching

## Docker Usage

### Building the Image
```bash
docker build -t round1b-solution .
```

### Running the Container
```bash
docker run -v /path/to/input:/app/input -v /path/to/output:/app/output round1b-solution
```

### Input Format
Place PDF files in the input directory along with:
- `persona.txt` - User persona description
- `job.txt` - Job to be done description

Alternatively, set environment variables:
- `PERSONA` - User persona description
- `JOB_TO_BE_DONE` - Job requirements

### Output Format
The system generates:
- `result.json` - Main output with ranked sections
- `debug_info.json` - Detailed scoring information
- `summary_report.txt` - Human-readable summary
- `processing.log` - Execution logs

## Local Development

### Installation
```bash
pip install -r requirements.txt
```

### Running Locally
```bash
cd src
python main.py
```

## Output Schema

```json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "User persona description",
    "job_to_be_done": "Job requirements",
    "processing_timestamp": "2025-01-24T...",
    "total_sections_analyzed": 150,
    "top_sections_selected": 10,
    "persona_analysis": {
      "extracted_role": "researcher",
      "identified_domain": "academic",
      "expertise_level": "expert"
    }
  },
  "extracted_sections": [
    {
      "rank": 1,
      "relevance_score": 0.8542,
      "document_name": "research_paper.pdf",
      "section_title": "Methodology",
      "page_number": 5,
      "content": {
        "text": "Section content...",
        "word_count": 245
      }
    }
  ],
  "subsection_analysis": [
    {
      "section_title": "Methodology",
      "relevant_paragraphs": [
        {
          "text": "Most relevant paragraph...",
          "relevance_score": 0.92,
          "key_terms": ["analysis", "methodology"]
        }
      ]
    }
  ]
}
```

## Performance Characteristics

- **Processing Time**: <60 seconds for typical document sets
- **Memory Usage**: <1GB RAM
- **Model Size**: ~200MB total (sentence transformer + dependencies)
- **CPU Only**: No GPU requirements
- **Offline Operation**: No network dependencies after model download

## Scoring Algorithm

### Hybrid Relevance Score
```
Final Score = 0.6 × Semantic Score + 0.25 × Lexical Score + 0.15 × Title Score
```

### Semantic Similarity
- Uses sentence transformers to generate embeddings
- Computes cosine similarity between section content and persona/job context
- Fallback to word overlap if model unavailable

### Lexical Matching
- Weighted fuzzy matching of search terms
- Priority-based term weighting (high/medium/low priority)
- RapidFuzz for robust string matching

### Title Matching
- Direct and partial matching of titles against search terms
- Higher weight for title relevance

## Error Handling

- Graceful degradation if ML models fail to load
- Fallback scoring using lexical methods only
- Robust PDF processing with multiple format support
- Comprehensive logging and error reporting

## Constraints Compliance

✅ **CPU Only**: No GPU requirements  
✅ **Model Size**: <1GB total footprint  
✅ **Processing Time**: <60 seconds typical  
✅ **Offline**: No network dependencies  
✅ **Platform**: Linux/amd64 Docker support  

## Testing

The solution has been tested with:
- Academic research papers
- Business reports and presentations
- Technical documentation
- Mixed document types
- Various persona types (researcher, analyst, student, etc.)

## Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Solution: System falls back to lexical scoring only
   - Check internet connection during first run for model download

2. **Memory Issues**
   - Solution: Processes documents one at a time
   - Reduce batch size if needed

3. **PDF Processing Errors**
   - Solution: Robust error handling with detailed logging
   - Skips problematic files and continues processing

### Log Analysis
Check `processing.log` in the output directory for detailed execution information.

## Future Enhancements

- Multi-language support
- Custom domain vocabularies
- Advanced subsection analysis
- Interactive relevance tuning
- Batch processing optimizations