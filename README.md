# 📄 AI Resume Analyzer

A powerful AI-driven application that analyzes resumes against job descriptions using advanced semantic matching with HuggingFace's sentence-transformers model. Get accurate match percentages and insights to optimize your resume for specific job opportunities.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

- **🤖 Advanced AI Analysis**: Uses `sentence-transformers/all-mpnet-base-v2` for semantic similarity matching
- **📄 PDF Resume Parsing**: Extracts text from PDF resumes using PyMuPDF and pdfminer
- **🎯 Accurate Scoring**: Provides realistic match percentages (10-95%) based on semantic analysis
- **⚡ Fast Processing**: Cached model loading for quick subsequent analyses
- **🎨 Clean UI**: Simple, intuitive Streamlit interface
- **🔄 Fallback System**: Keyword-based analysis if semantic analysis fails
- **📊 Detailed Insights**: Score breakdown and analysis details
- **🆓 Free to Use**: No API keys required, runs completely offline after initial setup

## 🚀 Demo

1. **Upload Resume**: Choose your PDF resume file
2. **Enter Job Description**: Paste the complete job description
3. **Analyze**: Click the analyze button to get your match score
4. **Get Results**: View your percentage match with color-coded feedback

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI/ML**: HuggingFace Sentence Transformers (all-mpnet-base-v2)
- **PDF Processing**: PyMuPDF, pdfminer.six
- **Data Processing**: NumPy, scikit-learn
- **Backend**: Python 3.8+

## 📋 Prerequisites

- Python 3.8 or higher
- Internet connection (for initial model download only)
- 2GB+ RAM (for model loading)
- 1GB+ free disk space (for model storage)

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/resume-analyzer.git
cd resume-analyzer
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment (Optional)
```bash
cp .env.example .env
# Edit .env file if needed
```

## 🚀 Usage

### Run the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### First Time Setup
- On first run, the app will download the sentence-transformers model (~400MB)
- This is a one-time download and will be cached for future use
- Ensure you have a stable internet connection for the initial setup

### Using the Analyzer
1. **Upload Resume**: Click "Choose a PDF file" and select your resume
2. **Enter Job Description**: Paste the complete job description in the text area
3. **Analyze**: Click "🔍 Analyze with AI" button
4. **View Results**: Get your match percentage with detailed feedback

