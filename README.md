# PDF Topic Analysis Tool 🔍

## Overview
A Python-based tool that performs topic analysis on PDF documents using Natural Language Processing (NLP) and Latent Dirichlet Allocation (LDA). The tool automatically detects the document's subject area and provides detailed topic distribution analysis.

## ✨ Key Features
- 📄 Smart PDF Text Extraction with Progress Tracking
- 🔍 Advanced Text Preprocessing (tokenization, stopwords removal)
- 📊 Topic Modeling with LDA
- 🎯 Automatic Subject Classification
- 📈 Topic Distribution Analysis
- 🎨 Color-coded Console Output

## 🚀 Quick Start
```bash
# Clone the repository
git clone https://github.com/KhaledAbdElazem/NLP_Project_0.0.2.git

# Install dependencies
pip install -r requirements.txt

# Run the application
python NLP_Project.py
```

## 📋 Requirements
```txt
pandas
numpy
scikit-learn
nltk
colorama
tabulate
PyPDF2
tkinter (usually comes with Python)
```

## 💻 Usage
Simply run the script and use the file dialog to select your PDF:
```python
python NLP_Project.py
```

## 🎯 Subject Categories
The tool can automatically classify documents into these categories:
- Software/Technology
- Business/Finance
- Health/Fitness
- Education/Academic
- Legal
- Religion/Spirituality
- Romance/Relationships
- General/Other

## 🛠️ Core Functions

### PDF Processing
```python
def read_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file with progress tracking"""

def preprocess_text(text: str) -> str:
    """Clean and tokenize text, remove stopwords"""
```

### Topic Modeling
```python
def train_lda_model(texts: List[str], n_topics: int = 7) -> Tuple[LDAModel, CountVectorizer, ndarray]:
    """Train LDA model with optimized parameters"""

def print_topics(model: LDAModel, feature_names: List[str], n_top_words: int = 10):
    """Display topics in a formatted table"""
```

### Subject Classification
```python
def determine_subject(top_words_list: List[List[str]]) -> str:
    """Determine document subject based on topic keywords"""
```

## 📊 Output Example
The tool provides:
- Color-coded topic analysis
- Top words for each topic
- Overall document subject
- Topic probability distribution
- Progress tracking for large PDFs

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.

## 📬 Contact & Support
- **Author**: [Khaled Abdelazem]
- **Email**: khaledabdelazem.work@gmail.com
- **Project Link**: https://github.com/KhaledAbdElazem/NLP_Project_0.0.2.git

## 🙏 Acknowledgements
- [NLTK](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/)
- [PyPDF2](https://pythonhosted.org/PyPDF2/)
<<<<<<< HEAD
- [Colorama](https://pypi.org/project/colorama/)
=======
- [Colorama](https://pypi.org/project/colorama/)
>>>>>>> 63da3d6b19836d772a1b0256e8c44d5e24e71f0e
