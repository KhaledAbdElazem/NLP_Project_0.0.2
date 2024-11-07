from utils import SUBJECT_KEYWORDS

def determine_subject(top_words_list):
    """Determine the general subject based on topic keywords"""
    all_words = set([word.lower() for topic_words in top_words_list for word in topic_words])
    
    subject_matches = {}
    for subject, keywords in SUBJECT_KEYWORDS.items():
        exact_matches = len(keywords.intersection(all_words))
        partial_matches = sum(1 for word in all_words 
                            for keyword in keywords 
                            if keyword in word and word != keyword)
        score = (exact_matches * 3) + partial_matches
        subject_matches[subject] = score
    
    best_match = max(subject_matches.items(), key=lambda x: x[1])
    
    if best_match[1] >= 1:
        return best_match[0]
    return "General/Other" 