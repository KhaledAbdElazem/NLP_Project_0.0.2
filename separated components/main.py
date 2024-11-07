import tkinter as tk
from tkinter import filedialog
from colorama import init, Fore, Style
import os
from tabulate import tabulate

from pdf_handler import read_pdf
from topic_analyzer import train_lda_model, print_topics
from subject_classifier import determine_subject

# Initialize colorama
init()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    print(Fore.MAGENTA + "\n=== PDF Topic Analysis ===" + Style.RESET_ALL)
    
    print(f"{Fore.CYAN}Please select a PDF file...{Style.RESET_ALL}")
    pdf_path = filedialog.askopenfilename(
        title="Select PDF file",
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
    )
    
    if not pdf_path:
        print(f"{Fore.RED}No file selected. Exiting...{Style.RESET_ALL}")
        exit()
    
    print(f"{Fore.CYAN}Reading PDF file: {os.path.basename(pdf_path)}{Style.RESET_ALL}")
    pdf_text = read_pdf(pdf_path)
    
    if pdf_text:
        min_chunk_size = 1000
        chunk_size = 2000
        texts = []
        
        for i in range(0, len(pdf_text), chunk_size):
            chunk = pdf_text[i:i+chunk_size]
            if len(chunk) >= min_chunk_size:
                texts.append(chunk)
        
        lda_model, vectorizer, lda_output = train_lda_model(texts, n_topics=7)
        
        print(Fore.BLUE + "\nMain topics in the PDF:" + Style.RESET_ALL)
        print_topics(lda_model, vectorizer.get_feature_names_out())
        
        top_words_list = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-10-1:-1]]
            top_words_list.append(top_words)
        
        subject = determine_subject(top_words_list)
        print(f"\n{Fore.BLUE}Document Subject: {Fore.GREEN}{subject}{Style.RESET_ALL}")
        
        print(Fore.BLUE + "\nOverall topic distribution:" + Style.RESET_ALL)
        avg_topic_distribution = lda_output.mean(axis=0)
        distribution_table = []
        for idx, topic_prob in enumerate(avg_topic_distribution):
            distribution_table.append([f"Topic {idx + 1}", f"{topic_prob:.3f}"])
        print(tabulate(distribution_table, headers=["Topic", "Probability"], tablefmt="grid"))
        
        input("\nPress Enter to exit...") 