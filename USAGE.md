# Round 1B: Interactive Persona-Driven Document Intelligence

## 🚀 How to Use

### **Option 1: Interactive Mode (Recommended)**
```bash
cd Round1b/src
python main.py
```

The system will ask you for:
1. **📄 PDF file path** - Provide path to your PDF (will be copied to `pdf/` folder)
2. **👤 Your persona** - Describe yourself (role, expertise, experience)
3. **🎯 Your task** - What you're looking for or need to accomplish

### **Option 2: Use Existing Files**
If you don't provide new inputs, the system uses existing files:
- `Round1b/input/persona.txt` - Your saved persona description
- `Round1b/input/job.txt` - Your saved job/task description  
- `Round1b/pdf/*.pdf` - Any PDF files in the pdf folder

### **Directory Structure**
```
Round1b/
├── pdf/                     # 📁 PDF files (auto-managed)
├── input/                   # 📝 Text inputs (persona & job)
│   ├── persona.txt         # 👤 Your description
│   └── job.txt            # 🎯 Your task
├── src/                   # 💻 Source code
│   └── main.py           # 🚀 Run this file
├── output/               # 📊 Results go here
└── USAGE.md             # 📖 This file
```

## 🎯 Example Interaction

```
🎯 PERSONA-DRIVEN DOCUMENT INTELLIGENCE
========================================
Please provide the following information:
(Press Enter to skip and use existing files if available)

1️⃣ PDF Document:
   Enter path to PDF file (or press Enter to skip): C:\MyPaper.pdf
   ✅ PDF copied to: Round1b\pdf\MyPaper.pdf

2️⃣ Your Persona:
   Describe yourself (role, expertise, experience, interests)
   Enter persona description: I am a software engineer specializing in AI...
   ✅ Persona saved to: Round1b\input\persona.txt

3️⃣ Your Task/Job:
   Describe what you're looking for or need to accomplish  
   Enter job description: I need to understand neural network architectures...
   ✅ Job description saved to: Round1b\input\job.txt

🎉 SUCCESS! Analysis completed in 15.2 seconds
📊 Found 10 relevant sections
📁 Results saved to: Round1b\output
```

## ⚠️ Error Handling

- **"NO PDF is available"** → No PDFs in the `pdf/` folder
- **"No data available"** → No persona/job descriptions found anywhere
- **Invalid PDF path** → File doesn't exist or isn't a PDF

## 📊 Output Files

Results are saved to `Round1b/output/`:
- **`result.json`** - Main structured output with ranked sections
- **`summary_report.txt`** - Human-readable summary
- **`debug_info.json`** - Detailed scoring and analysis information

## 🔄 Workflow

1. **Collect Inputs** → Interactive prompts or existing files
2. **Validate Data** → Check if PDF and descriptions are available  
3. **Process PDFs** → Extract text and structure from documents
4. **Analyze Persona** → Understand your background and needs
5. **Rank Sections** → Score relevance using hybrid AI+lexical matching
6. **Generate Output** → Create structured results tailored to you

## ✨ Features

- 🎯 **Interactive Input Collection** - Guides you through setup
- � **Smart File Management** - Auto-organizes PDFs and inputs
- 🔄 **Fallback to Existing Data** - Uses saved files if available
- 🚫 **Comprehensive Error Handling** - Clear messages for missing data
- 🧠 **Persona-Aware Analysis** - Adapts to your expertise level and domain
- � **Multiple Output Formats** - JSON, text, and debug information
