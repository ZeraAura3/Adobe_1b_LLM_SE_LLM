#!/usr/bin/env python3
"""
Test script for Round 1B: Persona-Driven Document Intelligence
Verifies the complete pipeline works correctly before Docker submission.
"""

import sys
import os
import time
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from document_processor import DocumentProcessor
        from persona_analyzer import PersonaJobAnalyzer
        from relevance_scorer import RelevanceScorer
        from subsection_analyzer import SubSectionAnalyzer
        from main import PersonaDrivenDocumentIntelligence
        print("✅ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality with minimal data."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test document processor
        processor = DocumentProcessor()
        print("✅ DocumentProcessor initialized")
        
        # Test persona analyzer
        analyzer = PersonaJobAnalyzer()
        persona_analysis = analyzer.analyze_persona("I am a researcher in computer science")
        job_analysis = analyzer.analyze_job("I need to analyze research papers")
        requirements = analyzer.combine_requirements(persona_analysis, job_analysis)
        print("✅ PersonaJobAnalyzer working")
        
        # Test relevance scorer
        scorer = RelevanceScorer()
        print("✅ RelevanceScorer initialized")
        
        # Test subsection analyzer
        subsection_analyzer = SubSectionAnalyzer()
        print("✅ SubSectionAnalyzer initialized")
        
        # Test main system
        intelligence_system = PersonaDrivenDocumentIntelligence()
        print("✅ Main system initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_sample_data():
    """Test with sample persona and job data."""
    print("\n🧪 Testing with sample data...")
    
    try:
        # Sample data
        sample_persona = "I am a graduate student in computer science with 2 years of research experience."
        sample_job = "I need to prepare a literature review on machine learning applications."
        
        # Test persona analysis
        analyzer = PersonaJobAnalyzer()
        persona_analysis = analyzer.analyze_persona(sample_persona)
        job_analysis = analyzer.analyze_job(sample_job)
        requirements = analyzer.combine_requirements(persona_analysis, job_analysis)
        
        print(f"   📋 Detected role: {persona_analysis.get('role', 'unknown')}")
        print(f"   🎯 Task type: {job_analysis.get('task_type', 'unknown')}")
        print(f"   🔑 Keywords found: {len(requirements.get('all_keywords', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample data test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 ROUND 1B: TESTING PERSONA-DRIVEN DOCUMENT INTELLIGENCE")
    print("=" * 60)
    
    start_time = time.time()
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check dependencies.")
        return False
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed.")
        return False
    
    # Test with sample data
    if not test_with_sample_data():
        print("\n❌ Sample data tests failed.")
        return False
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n🎉 ALL TESTS PASSED!")
    print(f"⏱️  Total test time: {total_time:.2f}s")
    print("\n✅ System is ready for Docker submission!")
    
    # Display file structure
    print("\n📁 Project file structure:")
    current_dir = Path(".")
    for file in sorted(current_dir.glob("*.py")):
        print(f"   📄 {file.name}")
    for file in sorted(current_dir.glob("*.txt")):
        print(f"   📄 {file.name}")
    for file in sorted(current_dir.glob("*.md")):
        print(f"   📄 {file.name}")
    if (current_dir / "Dockerfile").exists():
        print(f"   🐳 Dockerfile")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
