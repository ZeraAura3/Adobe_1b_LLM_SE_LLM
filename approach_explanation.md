# Round 1B: Persona-Driven Document Intelligence

## Approach Explanation

### Overview
This solution implements a multi-stage pipeline for persona-driven document intelligence that extracts and prioritizes relevant sections from PDF collections based on user persona and job requirements.

### Methodology

#### 1. Document Processing
- **PDF Extraction**: Uses `pdfplumber` for robust text extraction with structure preservation
- **Section Detection**: Employs heuristic-based algorithms to identify section headers using:
  - Numbered patterns (1., 1.1, I., A.)
  - Typography patterns (ALL CAPS, Title Case)
  - Keyword-based detection (Introduction, Methodology, etc.)
- **Text Structuring**: Organizes content into hierarchical sections with metadata

#### 2. Persona & Job Analysis
- **Role Detection**: Uses regex patterns to identify persona roles and experience levels
- **Domain Classification**: Maps personas to domain-specific keyword vocabularies
- **Task Type Identification**: Categorizes jobs (literature review, analysis, summarization)
- **Requirement Extraction**: Combines NLP techniques to extract key concepts and priorities

#### 3. Relevance Scoring
- **TF-IDF Vectorization**: Creates document-term matrices for semantic similarity
- **Cosine Similarity**: Measures alignment between sections and requirements
- **Multi-factor Scoring**: Incorporates:
  - Content relevance (keyword matching)
  - Section length optimization
  - Domain-specific bonuses
  - Title relevance weighting

#### 4. Sub-section Analysis
- **Granular Extraction**: Multiple strategies for sub-section identification:
  - Paragraph-based splitting
  - Sentence grouping for coherence
  - Key passage extraction for dense content
- **Quality Filtering**: Ensures meaningful content length and completeness
- **Relevance Ranking**: Scores sub-sections using weighted keyword density

### Technical Implementation

#### Models & Libraries Used
- **NLP**: spaCy (en_core_web_sm, ~50MB) for named entity recognition and noun phrase extraction
- **Text Processing**: NLTK for tokenization and POS tagging
- **Machine Learning**: scikit-learn TF-IDF vectorizer with cosine similarity
- **PDF Processing**: pdfplumber for reliable text extraction
- **Fallback Mechanisms**: Keyword-based scoring when ML libraries unavailable

#### Performance Optimizations
- **CPU-Only**: All processing designed for CPU execution without GPU dependencies
- **Memory Efficient**: Limited TF-IDF features (1000 max) and streaming processing
- **Time Constraints**: Optimized for <60s processing of 3-5 documents
- **Offline Operation**: No internet dependencies, all models pre-downloaded

#### Output Format
Generates structured JSON with:
- **Metadata**: Processing details, persona analysis, job analysis
- **Extracted Sections**: Top-ranked sections with relevance scores and factors
- **Sub-section Analysis**: Granular content with refined text and rankings

### Generic Solution Design
The system generalizes across domains through:
- **Dynamic Keyword Vocabularies**: Domain-agnostic keyword extraction
- **Flexible Section Detection**: Multiple heuristics for varied document formats
- **Adaptive Scoring**: Multi-factor relevance assessment
- **Persona-Job Alignment**: Context-aware requirement understanding

This approach ensures robust performance across academic, business, and educational document collections while maintaining the required performance constraints.
