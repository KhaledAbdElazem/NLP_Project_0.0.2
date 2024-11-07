#!/usr/bin/env python3
# This shebang line specifies that the script should be executed with Python 3.

# Import necessary libraries

# pandas: Used for data manipulation and analysis, commonly for handling tabular data.
import pandas as pd

# numpy: Used for efficient numerical operations and handling arrays, especially with matrices.
import numpy as np

# CountVectorizer: Converts text documents into a matrix of token counts, which serves as input for text analysis.
from sklearn.feature_extraction.text import CountVectorizer

# LatentDirichletAllocation: Implements LDA algorithm for topic modeling, which identifies topics within text.
from sklearn.decomposition import LatentDirichletAllocation

# train_test_split: Used to split data into training and testing sets, essential in machine learning workflows.
from sklearn.model_selection import train_test_split

# nltk: Natural Language Toolkit for handling and analyzing natural language text.
import nltk

# stopwords: Provides a set of common words that don’t carry significant meaning (e.g., “the”, “is”) for filtering.
from nltk.corpus import stopwords

# word_tokenize: Tokenizes or splits text into individual words.
from nltk.tokenize import word_tokenize

# re: Regular expressions module, allowing pattern matching in text, useful for text cleaning.
import re

# colorama: Allows for colored output in the terminal, enhancing readability for messages.
from colorama import init, Fore, Back, Style

# tabulate: Formats data into tables for console output, which improves readability.
from tabulate import tabulate

# PyPDF2: Library to read and extract text from PDF files.
import PyPDF2

# os: Provides operating system dependent functions, such as accessing file paths.
import os

# tkinter: Library for creating graphical interfaces. Used here to open a file selection dialog.
from tkinter import filedialog
import tkinter as tk

# Initialize colorama, allowing for colored text output in different operating systems.
init()

# Download necessary NLTK datasets
nltk.download('punkt')    # 'punkt' is needed for word tokenization.
nltk.download('stopwords') # Download stopwords list for filtering common words.

def preprocess_text(text):
    """
    Preprocesses the input text by performing several cleaning operations.
    
    Args:
        text (str): Raw input text
    
    Returns:
        str: Cleaned and preprocessed text
    """
    # Convert all characters in the text to lowercase to ensure uniformity
    text = text.lower()
    
    # Remove all characters that are not letters or spaces
    # `re.sub` replaces any character that isn't an uppercase/lowercase letter or space with an empty string.
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text into individual words (tokens) using `word_tokenize`.
    tokens = word_tokenize(text)
    
    # Load English stopwords from NLTK to filter out common words that add little meaning.
    stop_words = set(stopwords.words('english'))
    
    # Filter out stopwords from the tokens list.
    tokens = [token for token in tokens if token not in stop_words]
    
    # Combine the cleaned tokens back into a single string with spaces separating each word.
    return ' '.join(tokens)

def train_lda_model(texts, n_topics=5):
    """
    Trains an LDA model on preprocessed text data to identify topics.
    
    Args:
        texts (list of str): List of text documents to be analyzed.
        n_topics (int): Number of topics to be identified in the text corpus.
    
    Returns:
        lda_model: Trained LatentDirichletAllocation model.
        vectorizer: CountVectorizer object used to transform the text data.
        lda_output: Document-topic distribution matrix.
    """
    # Preprocess each text in the texts list by applying the preprocess_text function.
    print(Fore.CYAN + "\nPreprocessing texts..." + Style.RESET_ALL)
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Create a document-term matrix where each row is a document and each column is a token.
    print(Fore.CYAN + "Creating document-term matrix..." + Style.RESET_ALL)
    vectorizer = CountVectorizer(
        max_df=0.95,      # Ignores terms that appear in more than 95% of documents, reducing common words.
        min_df=2,         # Ignores terms that appear in fewer than 2 documents, reducing rare words.
        max_features=5000, # Limits vocabulary to the top 5000 most common terms for efficiency.
        stop_words='english' # Uses English stopwords to filter additional common words.
    )
    doc_term_matrix = vectorizer.fit_transform(processed_texts)
    
    # Train the LDA model with specified number of topics.
    print(Fore.CYAN + "Training LDA model..." + Style.RESET_ALL)
    lda_model = LatentDirichletAllocation(
        n_components=n_topics, # Sets the number of topics to discover.
        random_state=42,       # Ensures consistent results on repeated runs.
        learning_method='batch' # Uses batch learning to process the entire dataset at once.
    )
    
    # Fit the LDA model on the document-term matrix to generate topic distributions.
    lda_output = lda_model.fit_transform(doc_term_matrix)
    
    return lda_model, vectorizer, lda_output

def print_topics(model, feature_names, n_top_words=10):
    """
    Prints the top words for each topic identified by the LDA model.
    
    Args:
        model: Trained LDA model.
        feature_names (list): List of words corresponding to features in the document-term matrix.
        n_top_words (int): Number of top words to display per topic.
    """
    # List to store topics and their associated top words.
    topics_table = []
    
    # Iterate through each topic component of the model.
    for topic_idx, topic in enumerate(model.components_):
        # Select the top words associated with the topic based on the word weights.
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]
        
        # Format the words list into a comma-separated string.
        formatted_words = ', '.join(top_words)
        
        # Truncate strings that are too long (optional).
        if len(formatted_words) > 80:
            formatted_words = formatted_words[:77] + "..."
        
        # Append each topic's label and words to the topics table.
        topics_table.append([
            f"{Fore.GREEN}Topic {topic_idx + 1}{Style.RESET_ALL}",
            f"{Fore.YELLOW}{formatted_words}{Style.RESET_ALL}"
        ])
    
    # Print the topics table with headers and a grid format.
    print("\n" + tabulate(
        topics_table,
        headers=["Topic", "Top Words"],
        tablefmt="grid",
        maxcolwidths=[10, 80]
    ))

def read_pdf(pdf_path):
    """
    Extracts text content from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Display total pages and progress message for long PDFs.
            total_pages = len(pdf_reader.pages)
            print(f"{Fore.CYAN}Processing {total_pages} pages...{Style.RESET_ALL}")
            
            # Extract text from each page, showing progress every 50 pages.
            for i, page in enumerate(pdf_reader.pages):
                if (i + 1) % 50 == 0:
                    print(f"{Fore.CYAN}Processed {i + 1}/{total_pages} pages...{Style.RESET_ALL}")
                text += page.extract_text() + " "
        
        return text.strip()
    except Exception as e:
        print(f"{Fore.RED}Error reading PDF: {e}{Style.RESET_ALL}")
        return None

def determine_subject(top_words_list):
    """
    Determines the general subject based on the top words in topics.
    
    Args:
        top_words_list (list of lists): List of top words for each topic.
    
    Returns:
        str: Best-matching subject category.
    """
    # Define keywords associated with each possible subject category.
    subject_keywords = {
        'Software/Technology': {'software', 'programming', 'data', 'application', 'web', 'cloud'},
        'Business/Finance': {'business', 'financial', 'company', 'market', 'investment'},
        'Health/Fitness': {'health', 'body', 'exercise', 'diet', 'nutrition', 'medical'},
        'Education/Academic': {'education', 'student', 'learning', 'study', 'university'},
        'Legal': {'legal', 'law', 'court', 'compliance', 'contract'},
        'Religion/Spirituality': {'god', 'faith', 'spirit', 'church', 'religious'},
        'Romance/Relationships': {'love', 'heart', 'relationship', 'romance', 'marriage'}
    }
    
    # Flatten the top words into a single set.
    all_words = set([word.lower() for topic_words in top_words_list for word in topic_words])
    
    # Calculate a score for each subject based on keyword matches.
    subject_matches = {}
    for subject, keywords in subject_keywords.items():
        exact_matches = len(keywords.intersection(all_words))
        partial_matches = sum(1 for word in all_words if any(word.startswith(kw) for kw in keywords))
        subject_matches[subject] = exact_matches + 0.5 * partial_matches
    
    # Determine and return the subject with the highest score.
    best_subject = max(subject_matches, key=subject_matches.get)
    return best_subject

def main():
    """
    Main function that coordinates file selection, PDF text extraction, LDA training, 
    and displays the identified topics and suggested subject.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window to focus on file selection dialog.
    
    # Prompt the user to select a PDF file to analyze.
    pdf_path = filedialog.askopenfilename(title="Select a PDF file", filetypes=[("PDF files", "*.pdf")])
    if not pdf_path:
        print(Fore.RED + "No file selected." + Style.RESET_ALL)
        return
    
    # Extract text from the selected PDF.
    pdf_text = read_pdf(pdf_path)
    if not pdf_text:
        print(Fore.RED + "Failed to extract text from the PDF." + Style.RESET_ALL)
        return
    
    # Process extracted text by splitting it into sentences/documents for topic analysis.
    texts = pdf_text.split('. ')
    
    # Train the LDA model with the processed texts.
    lda_model, vectorizer, lda_output = train_lda_model(texts, n_topics=5)
    
    # Display the identified topics with top words per topic.
    print(Fore.CYAN + "\nIdentified Topics and Top Words:" + Style.RESET_ALL)
    print_topics(lda_model, vectorizer.get_feature_names_out())
    
    # Identify the general subject based on the top words in topics.
    top_words_list = [
        [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-6:-1]]
        for topic in lda_model.components_
    ]
    suggested_subject = determine_subject(top_words_list)
    
    # Display the determined subject.
    print(Fore.CYAN + f"\nSuggested Subject: {Fore.YELLOW}{suggested_subject}{Style.RESET_ALL}")

# Run the main function to execute the script.
if __name__ == "__main__":
    main()
