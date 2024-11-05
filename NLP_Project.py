#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from colorama import init, Fore, Back, Style
from tabulate import tabulate
import PyPDF2
import os
from tkinter import filedialog
import tkinter as tk

# Initialize colorama for cross-platform colored output
init()

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

# Example usage
def train_lda_model(texts, n_topics=5):
    # Preprocess texts
    print(Fore.CYAN + "\nPreprocessing texts..." + Style.RESET_ALL)
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Create document-term matrix with adjusted parameters for small datasets
    print(Fore.CYAN + "Creating document-term matrix..." + Style.RESET_ALL)
    vectorizer = CountVectorizer(
        max_df=0.95,      # Words appearing in >95% of docs are ignored
        min_df=2,         # Words appearing in <2 docs are ignored
        max_features=5000, # Limit vocabulary size
        stop_words='english'
    )
    doc_term_matrix = vectorizer.fit_transform(processed_texts)
    
    # Train LDA model
    print(Fore.CYAN + "Training LDA model..." + Style.RESET_ALL)
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method='batch'
    )
    
    # Fit the model
    lda_output = lda_model.fit_transform(doc_term_matrix)
    
    return lda_model, vectorizer, lda_output

def print_topics(model, feature_names, n_top_words=10):
    # Create a list to store topic information
    topics_table = []
    
    for topic_idx, topic in enumerate(model.components_):
        # Get the top words for this topic
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]
        # Format the words list more cleanly
        formatted_words = ', '.join(top_words)
        # Truncate long strings if necessary (optional)
        if len(formatted_words) > 80:
            formatted_words = formatted_words[:77] + "..."
            
        topics_table.append([
            f"{Fore.GREEN}Topic {topic_idx + 1}{Style.RESET_ALL}",
            f"{Fore.YELLOW}{formatted_words}{Style.RESET_ALL}"
        ])
    
    # Print the table with a single header
    print("\n" + tabulate(
        topics_table,
        headers=["Topic", "Top Words"],
        tablefmt="grid",
        maxcolwidths=[10, 80]  # Limit column widths
    ))

def read_pdf(pdf_path):
    """Extract text from a PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Add progress indicator for long PDFs
            total_pages = len(pdf_reader.pages)
            print(f"{Fore.CYAN}Processing {total_pages} pages...{Style.RESET_ALL}")
            
            for i, page in enumerate(pdf_reader.pages):
                # Show progress every 50 pages
                if (i + 1) % 50 == 0:
                    print(f"{Fore.CYAN}Processed {i + 1}/{total_pages} pages...{Style.RESET_ALL}")
                text += page.extract_text() + " "
        
        return text.strip()
    except Exception as e:
        print(f"{Fore.RED}Error reading PDF: {e}{Style.RESET_ALL}")
        return None

def determine_subject(top_words_list):
    """Determine the general subject based on topic keywords"""
    # Expanded dictionary of subjects and their related keywords
    subject_keywords = {
        'Software/Technology': {
            'software', 'programming', 'computer', 'data', 'system', 'code', 'application', 
            'digital', 'technology', 'web', 'network', 'internet', 'database', 'algorithm',
            'developer', 'hardware', 'security', 'cloud', 'interface', 'server'
        },
        'Business/Finance': {
            'business', 'market', 'financial', 'company', 'management', 'revenue', 'customer',
            'strategy', 'sales', 'profit', 'investment', 'stock', 'money', 'corporate',
            'economy', 'trade', 'industry', 'commercial', 'entrepreneur', 'retail'
        },
        'Health/Fitness': {
            'health', 'body', 'exercise', 'diet', 'weight', 'muscle', 'fat', 'fitness',
            'nutrition', 'protein', 'calories', 'workout', 'healthy', 'bodyfat', 'food',
            'training', 'strength', 'metabolism', 'supplements', 'gym', 'wellness',
            'medical', 'patient', 'treatment', 'disease', 'clinical', 'hospital',
            'doctor', 'medicine', 'symptoms'
        },
        'Education/Academic': {
            'education', 'student', 'learning', 'teaching', 'academic', 'school',
            'university', 'research', 'study', 'knowledge', 'professor', 'college',
            'classroom', 'curriculum', 'lecture', 'theory', 'science', 'thesis'
        },
        'Legal': {
            'legal', 'law', 'court', 'regulation', 'compliance', 'contract', 'rights',
            'policy', 'legislation', 'attorney', 'judicial', 'lawyer', 'plaintiff',
            'defendant', 'justice', 'statutory', 'constitutional'
        },
        'Religion/Spirituality': {
            'god', 'faith', 'spirit', 'holy', 'prayer', 'church', 'worship', 'divine',
            'religious', 'sacred', 'biblical', 'jesus', 'christian', 'bible', 'spiritual',
            'lord', 'soul', 'heaven', 'belief', 'prophet', 'salvation'
        },
        'Romance/Relationships': {
            'love', 'heart', 'romance', 'relationship', 'feeling', 'emotion', 'kiss',
            'passion', 'romantic', 'marriage', 'dating', 'couple', 'boyfriend', 'girlfriend',
            'wedding', 'affair', 'intimate', 'lover', 'partner', 'dating'
        }
    }
    
    # Flatten the list of top words and include words from topic distributions
    all_words = set([word.lower() for topic_words in top_words_list for word in topic_words])
    
    # Calculate matches for each subject with weighted scoring
    subject_matches = {}
    for subject, keywords in subject_keywords.items():
        # Count exact matches
        exact_matches = len(keywords.intersection(all_words))
        # Count partial matches (words containing keywords)
        partial_matches = sum(1 for word in all_words 
                            for keyword in keywords 
                            if keyword in word and word != keyword)
        # Weight exact matches more heavily than partial matches
        score = (exact_matches * 2) + partial_matches
        subject_matches[subject] = score
    
    # Find the subject with the most matches
    best_match = max(subject_matches.items(), key=lambda x: x[1])
    
    # Only return a subject if it has a minimum score
    if best_match[1] >= 2:  # Require at least 2 points to classify
        return best_match[0]
    return "General/Other"

# Example usage
if __name__ == "__main__":
    # Create and hide the tkinter root window
    root = tk.Tk()
    root.withdraw()

    print(Fore.MAGENTA + "\n=== PDF Topic Analysis ===" + Style.RESET_ALL)
    
    # Open file dialog for PDF selection
    print(f"{Fore.CYAN}Please select a PDF file...{Style.RESET_ALL}")
    pdf_path = filedialog.askopenfilename(
        title="Select PDF file",
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
    )
    
    if not pdf_path:
        print(f"{Fore.RED}No file selected. Exiting...{Style.RESET_ALL}")
        exit()
    
    # Read PDF content
    print(f"{Fore.CYAN}Reading PDF file: {os.path.basename(pdf_path)}{Style.RESET_ALL}")
    pdf_text = read_pdf(pdf_path)
    
    if pdf_text:
        # Split into more manageable chunks (2000 characters instead of 1000)
        # and ensure minimum chunk size to avoid too many small pieces
        min_chunk_size = 1000
        chunk_size = 2000
        texts = []
        
        for i in range(0, len(pdf_text), chunk_size):
            chunk = pdf_text[i:i+chunk_size]
            if len(chunk) >= min_chunk_size:
                texts.append(chunk)
        
        # Train the model with 7 topics instead of 3
        lda_model, vectorizer, lda_output = train_lda_model(texts, n_topics=7)
        
        # Print topics
        print(Fore.BLUE + "\nMain topics in the PDF:" + Style.RESET_ALL)
        print_topics(lda_model, vectorizer.get_feature_names_out())
        
        # Collect top words for subject determination
        top_words_list = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-10-1:-1]]
            top_words_list.append(top_words)
        
        # Determine and print the subject
        subject = determine_subject(top_words_list)
        print(f"\n{Fore.BLUE}Document Subject: {Fore.GREEN}{subject}{Style.RESET_ALL}")
        
        # Print overall topic distribution
        print(Fore.BLUE + "\nOverall topic distribution:" + Style.RESET_ALL)
        avg_topic_distribution = lda_output.mean(axis=0)
        distribution_table = []
        for idx, topic_prob in enumerate(avg_topic_distribution):
            distribution_table.append([f"Topic {idx + 1}", f"{topic_prob:.3f}"])
        print(tabulate(distribution_table, headers=["Topic", "Probability"], tablefmt="grid"))
        
        # Wait for user input before closing
        input("\nPress Enter to exit...")
