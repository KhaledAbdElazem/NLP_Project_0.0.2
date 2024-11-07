from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from colorama import Fore, Style
from tabulate import tabulate
from text_processor import preprocess_text

def train_lda_model(texts, n_topics=5):
    print(Fore.CYAN + "\nPreprocessing texts..." + Style.RESET_ALL)
    processed_texts = [preprocess_text(text) for text in texts]
    
    print(Fore.CYAN + "Creating document-term matrix..." + Style.RESET_ALL)
    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=5000,
        stop_words='english'
    )
    doc_term_matrix = vectorizer.fit_transform(processed_texts)
    
    print(Fore.CYAN + "Training LDA model..." + Style.RESET_ALL)
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method='batch'
    )
    
    lda_output = lda_model.fit_transform(doc_term_matrix)
    
    return lda_model, vectorizer, lda_output

def print_topics(model, feature_names, n_top_words=10):
    topics_table = []
    
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]
        formatted_words = ', '.join(top_words)
        if len(formatted_words) > 80:
            formatted_words = formatted_words[:77] + "..."
            
        topics_table.append([
            f"{Fore.GREEN}Topic {topic_idx + 1}{Style.RESET_ALL}",
            f"{Fore.YELLOW}{formatted_words}{Style.RESET_ALL}"
        ])
    
    print("\n" + tabulate(
        topics_table,
        headers=["Topic", "Top Words"],
        tablefmt="grid",
        maxcolwidths=[10, 80]
    )) 