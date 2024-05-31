import sys
import os
import pandas as pd
import spacy
import re
import numpy as np
import statistics
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import stanza
import string
from collections import Counter
from urllib.parse import urlparse
from nltk.corpus import stopwords
from datetime import datetime, timedelta
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
from sklearn.utils import shuffle
from util_analysis import *


# STEP 1: Defining helper functions
# Preprocessing helper functions
def preprocess_author(author):
    """
    Clean Author column of data to extract author information if it is provided as a URL.
    :param author
    :return: cleaned author information
    """
    if pd.notna(author) and author.startswith('http'):
        # Extract the last segment of the URL as the author name
        parsed_url = urlparse(author)
        path_segments = parsed_url.path.rstrip('/').split('/')
        
        # Check if the URL ends with 'index' and extract the name accordingly (this is a special case for German authors)
        if path_segments[-1] == 'index':
            return path_segments[-2].replace('_', ' ') if len(path_segments) > 2 else "Unknown"
        else:
            return path_segments[-1].replace('_', ' ') if path_segments[-1] else "Unknown"
    elif pd.isna(author) or str(author).strip().lower() == 'nan' or '2023' in str(author):  
        return "Unknown"
    else:
        return author

def clean_text_content(text):
    """
    Clean textual content from URLs, tags, newline characters and tabs, remove markup, numbers, emojis, special characters and repetitive phrases related to subscription.
    :text: input string
    :return cleaned text
    """
    # Remove URLs from text
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www.\S+', '', text)
    # Remove tags like @john_smith
    text = re.sub(r'@\S+', '', text)
    # Remove newline characters and tabs
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    # Remove italics markup
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Remove bold markup
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove emojis
    RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
    text = RE_EMOJI.sub(r'', text)
    # Remove other special characters and repetitive phrases
    special_chars_and_phrases = ['"', '–', "'",'„', '“', '%', '$', '”', '(', ')', ',', '’', '...', '—', '…', '  ', '[', ']', '..', 'Stand::', '€', 'SPIEGEL+', ':', '©', '»', '«', '/ Woche / Monat / Monat', 'Schon DigitalAbonnent? Hier anmelden']
    for element in special_chars_and_phrases:
        text = text.replace(element, '')
    return text

def preprocess_dataframe(df):
    """
    Preprocess the 'Author' column in a DataFrame
    :param df: Input DataFrame with a column 'Author'.
    :return: DataFrame with the preprocessed 'Author' column.
    """
    specific_newspaper_names = ['Der Tagesspiegel', 'DER SPIEGEL', 'tagesschau.de', 'BLZ', 'dpa,BLZ']

    # Convert 'Author' column to strings to handle NaN values
    df['Author'] = df['Author'].astype(str)

    # Remove specific names of newspapers as authors
    df['Author'] = df['Author'].apply(lambda x: "Unknown" if x.strip() in specific_newspaper_names else x)

    # Extract author names from URLs and replace empty or whitespace authors with "Unknown"
    df['Author'] = df['Author'].apply(preprocess_author)
    
    # Preprocess textual content of each article
    df['Text'] = df['Text'].apply(clean_text_content)

    return df

# -------------------------------
# Basic analysis helper functions
def calculate_author_ratios(frequencies):
    """
    Calculate the ratios of unknown and known authors based on the given article frequencies.
    :param frequencies (dict): A dictionary containing the frequency of articles attributed to different authors
    :return: tuple of two floats representing ratios
    """
    total_articles = sum(frequencies.values())
    unknown_authors = frequencies['Unknown']
    known_authors = total_articles - unknown_authors
    ratio_unknown = unknown_authors / total_articles
    ratio_known = known_authors / total_articles
    return ratio_unknown, ratio_known
    
def get_frequencies(df, language):
    """
    Get token, lemma, POS, and NER frequencies for a given language using a Stanza pipeline.
    :param df: DataFrame with a column 'Text' containing the text data.
    :param language: ISO language code ('en' for English, 'de' for German).
    :return: tuple of token, lemma, POS, and NER frequencies.
    """
    # Load Stanza pipeline for the specified language
    nlp = stanza.Pipeline(language)
    
    other_phrases = ["’s", "n’t", "’re", "'s", "I"]

    # Get stop words for the specified language
    if language == 'en':
        stop_words = stopwords.words('english')
        stop_words.extend(other_phrases)
    elif language == 'de':
        stop_words = stopwords.words('german')
        stop_words.extend(other_phrases)

    # Initialize counters and lists
    token_frequencies = Counter()
    lemma_frequencies = Counter()
    pos_frequencies = Counter()
    ner_frequencies = Counter()

    # Process articles with Stanza
    for article in df["Text"]:
        processed_article = nlp(article)
        sentences = processed_article.sentences
        
        for sentence in sentences:
            tokens = [token.text for token in sentence.words if token.text.lower() not in stop_words and token.text not in string.punctuation]
            lemmas = [token.lemma for token in sentence.words if token.lemma not in stop_words and token.text not in string.punctuation]
            pos = [word.pos for word in sentence.words if word.lemma not in stop_words and word.text not in string.punctuation]
            ner = [token.ner for token in sentence.tokens if token.text not in stop_words]
    
            token_frequencies.update(tokens)
            lemma_frequencies.update(lemmas)
            pos_frequencies.update(pos)
            ner_frequencies.update(ner)
            
    return token_frequencies, lemma_frequencies, pos_frequencies, ner_frequencies

def calculate_title_statistics(data, nlp_pipeline):
    '''
    Calculate statistics about the titles of articles.
    :param data: DataFrame containing article data.
    :param nlp_pipeline: pipeline for tokenization.
    :return: Tuple of title statistics (mean, std, median, mode, variance, min_length, max_length).
    '''
    data["Title Tokens"] = data["Title"].apply(lambda title: tokenize_title(title, nlp_pipeline))

    word_lengths = data["Title Tokens"].apply(len)
    mean_words = round(statistics.mean(word_lengths))
    std_words = round(statistics.stdev(word_lengths))
    median_words = round(statistics.median(word_lengths))
    mode_words = statistics.mode(word_lengths)
    variance_words = round(statistics.variance(word_lengths))
    min_length_words = min(word_lengths)
    max_length_words = max(word_lengths)
    return mean_words, std_words, median_words, mode_words, variance_words, min_length_words, max_length_words

def calculate_type_token_ratio(token_frequencies):
    '''
    Calculate type-token ratio based on token frequencies.
    :param token_frequencies: Dictionary containing token frequencies.
    :return: Tuple of type count, token count, and type-token ratio.
    '''
    num_types = len(token_frequencies.keys())
    num_tokens = sum(token_frequencies.values())
    tt_ratio = num_types / num_tokens
    
    return num_types, num_tokens, tt_ratio

def calculate_ttr_confidence_interval(token_frequencies, confidence=0.95):
    '''
    Calculate the confidence interval for Type-Token Ratio (TTR).
    :param token_frequencies: Dictionary containing token frequencies.
    :param confidence: Confidence level for the interval, defaults to 0.95.
    :return: Tuple representing the lower and upper bounds of the confidence interval for TTR.
    '''
    num_types, num_tokens, tt_ratio = calculate_type_token_ratio(token_frequencies)
    
    # Calculate standard deviation for TTR
    ttr_std_dev = np.sqrt(tt_ratio * (1 - tt_ratio) / num_tokens)

    # Calculate confidence interval
    margin_of_error_ttr = stats.t.ppf((1 + confidence) / 2, num_tokens - 1) * ttr_std_dev / np.sqrt(num_tokens)
    lower_bound_ttr = tt_ratio - margin_of_error_ttr
    upper_bound_ttr = tt_ratio + margin_of_error_ttr
    
    return lower_bound_ttr, upper_bound_ttr

# I read about randomized testing here: https://softwareengineering.stackexchange.com/questions/147134/how-should-i-test-randomness
def randomization_test_ttr(group_a_token_frequencies, group_b_token_frequencies, num_iterations=1000):
    '''
    Perform a randomization test to compare Type-Token Ratios (TTR) between English and German.
    :param group_a_token_frequencies: Token frequencies for language A.
    :param group_b_token_frequencies: Token frequencies for language B.
    :param num_iterations: Number of iterations for the randomization test, defaults to 1000.
    :return: P-value indicating the likelihood of observing the difference in TTR between the groups.
    '''
    (_, _, observed_difference_ttr_a), (_, _, observed_difference_ttr_b) = calculate_type_token_ratio(group_a_token_frequencies), calculate_type_token_ratio(group_b_token_frequencies)
    observed_difference_ttr = observed_difference_ttr_a - observed_difference_ttr_b
    
    combined_tokens = list(group_a_token_frequencies.keys()) + list(group_b_token_frequencies.keys())

    differences_ttr = []
    for _ in range(num_iterations):
        shuffled_tokens = shuffle(combined_tokens)
        new_group_a_tokens = shuffled_tokens[:len(group_a_token_frequencies)]
        new_group_b_tokens = shuffled_tokens[len(group_a_token_frequencies):]

        # Calculate TTR for the shuffled groups
        new_group_a_ttr = calculate_type_token_ratio(Counter(new_group_a_tokens))[2]
        new_group_b_ttr = calculate_type_token_ratio(Counter(new_group_b_tokens))[2]

        differences_ttr.append(new_group_a_ttr - new_group_b_ttr)

    p_value_ttr = (np.abs(differences_ttr) >= np.abs(observed_difference_ttr)).mean()
    return p_value_ttr

# -------------------------------
# NERC helper functions
def extract_countries(text, nlp_model, special_cases=None):
    '''
    Extract unique location names from article content using SpaCy.
    :param text: Text containing location entities.
    :param nlp_model: SpaCy for language processing.
    :param special_cases: Dictionary of special cases to handle specific entity variations (e.g. US and U.S.).
    :return: Set of unique location names mentioned in the text.
    '''
    doc = nlp_model(text)
    unique_loc_names_per_article = set()

    for ent in doc.ents:
    # Check if the entity is a location (LOC) or geopolitical entity (GPE)
        if ent.label_ == "LOC" or ent.label_ == "GPE":
            if special_cases and ent.text.lower() in special_cases: 
                unique_loc_names_per_article.add(special_cases[ent.text.lower()])
            else:
                unique_loc_names_per_article.add(ent.text)

    return unique_loc_names_per_article

def extract_countries_and_domains(text, url, nlp_model, special_cases=None):
    '''
    Extract unique location names and domain names using SpaCy.
    :param text: Text containing location entities.
    :param url: URL of the article for extracting the domain.
    :param nlp_model: SpaCy for language processing.
    :param special_cases: Dictionary of special cases to handle specific entity variations (e.g. US and U.S.).
    :return: Tuple containing the domain and a set of unique location names mentioned in the text.
    '''
    doc = nlp_model(text)
    unique_loc_names_per_article = set()

    for ent in doc.ents:
        if ent.label_ == "LOC" or ent.label_ == "GPE":
        # Check if the entity is a location (LOC) or geopolitical entity (GPE)
            if special_cases and ent.text.lower() in special_cases:
                unique_loc_names_per_article.add(special_cases[ent.text.lower()])
            else:
                unique_loc_names_per_article.add(ent.text)

    # Extract domain name from URL
    domain = urlparse(url).hostname.replace('www.', '')

    return domain, unique_loc_names_per_article

# -------------------------------
# Clustering helper functions
def get_lemmas(article, stop_words, nlp):
    '''
    Get lemmas from an article.
    :param article: Article text.
    :param stop_words: List of stop words.
    :param nlp: NLP pipeline.
    :return: List of lemmatized words.
    '''
    processed_article = nlp.process(article)
    all_lemmas = []
    other_punct_marks_and_phrases = ["’s", "n’t", "’re", "'s", "I", "say", "sagen", "said"]

    stop_words.extend(other_punct_marks_and_phrases)

    for s in processed_article.sentences:
        lemmas = [word.lemma.lower() for word in s.words if word.lemma is not None]
        clean_lemmas = [lemma for lemma in lemmas if lemma not in stop_words and lemma not in string.punctuation]
        all_lemmas.extend(clean_lemmas)

    return all_lemmas

def get_tfidf_representation(articles, tokenizer_func, model_path, num_keywords=10):
    '''
    Get TF-IDF representation of articles.
    :param articles: List of article texts.
    :param tokenizer_func: Tokenizer function.
    :param model_path: Path to the word2vec model.
    :param num_keywords: Number of keywords to extract.
    :return: tuple of document representations and extracted keywords.
    '''
    # Initialise TF-IDF vectoriser
    vectorizer = TfidfVectorizer(use_idf=True, tokenizer=tokenizer_func)
    # Transform articles into TF-IDF representations
    tf_idf = vectorizer.fit_transform(articles)
    all_terms = vectorizer.get_feature_names_out()

    print("loading word2vec") 
    model = KeyedVectors.load_word2vec_format(model_path)
    print("done loading word2vec")

    # Extract top keywords for each document based on TF-IDF scores
    keywords = []
    for i in range(tf_idf.shape[0]):
        row = np.squeeze(tf_idf[i].toarray())
        top_terms, _ = get_top_tfidf_features(row, all_terms, top_n=num_keywords)
        keywords.append(top_terms)

    # Calculate document representations as the mean word vectors of extracted keywords
    doc_representations = []
    for doc_keywords in keywords:
        doc_representation = []
        for keyword in doc_keywords:
            try:
                word_representation = model[keyword]
                doc_representation.append(word_representation)
            except KeyError:
                pass

        mean_keywords = np.mean(doc_representation, axis=0)
        doc_representations.append(mean_keywords)

    return doc_representations, keywords

def cluster_articles(doc_representations, num_clusters):
    '''
    Cluster articles based on document representations.
    :param doc_representations: List of document representations.
    :param num_clusters: Number of clusters.
    :return: List of cluster labels.
    '''
    # Apply KMeans clustering to document representations
    km = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    km.fit(doc_representations)
    # Get cluster labels for each article
    clusters = km.labels_.tolist()
    return clusters

def print_cluster_top_keywords(keywords, clusters, num_clusters, top_n=30):
    '''
    Print top keywords for each cluster.
    :param keywords: List of keyword lists for each document.
    :param clusters: List of cluster labels for each document.
    :param num_clusters: Number of clusters.
    :param top_n: Number of top keywords to display.
    '''
    for cluster_id in range(num_clusters):
        # Extract keywords for articles in the current cluster
        cluster_keywords = [keyword for i, keyword_list in enumerate(keywords) if clusters[i] == cluster_id for keyword in keyword_list]
        # Get top keywords for the cluster
        top_keywords = Counter(cluster_keywords).most_common(top_n)
        
        print(f"Cluster {cluster_id} Top Keywords:")
        print(top_keywords)
        print()
        
def generate_cluster_wordcloud(keywords, clusters, num_clusters, top_n=25):
    '''
    Generate word cloud for each cluster.
    :param keywords: List of keyword lists for each document.
    :param clusters: List of cluster labels for each document.
    :param num_clusters: Number of clusters.
    :param top_n: Number of top keywords to include in the word cloud.
    '''
    # Create a subplot grid with the specified number of columns
    fig, axs = plt.subplots(1, num_clusters, figsize=(15, 8))

    for cluster_id in range(num_clusters):
        # Extract keywords for articles in the current cluster
        cluster_keywords = [keyword for i, keyword_list in enumerate(keywords) if clusters[i] == cluster_id for keyword in keyword_list]
        top_keywords = Counter(cluster_keywords).most_common(top_n)

        # Create a word cloud for each cluster
        wordcloud = WordCloud(width=400, height=200, background_color='white').generate_from_frequencies(dict(top_keywords))

        # Plot the word cloud for the current cluster in the corresponding subplot
        axs[cluster_id].imshow(wordcloud, interpolation='bilinear')
        axs[cluster_id].axis('off')
        axs[cluster_id].set_title(f'Cluster {cluster_id}')

    plt.tight_layout()
    plt.show()
    
def analyze_domain_distribution(overview, content, num_clusters):
    '''
    Analyze domain distribution within each cluster.
    :param overview: DataFrame containing cluster overview.
    :param content: DataFrame containing article content.
    :param num_clusters: Number of clusters.
    :return: Dictionary mapping domains to colors.
    '''
    # Create subplots for each cluster
    fig, axes = plt.subplots(1, num_clusters, figsize=(20, 5))
    domain_color_map = {}
    # Extract unique domains from URLs
    all_domains = content['URL'].apply(lambda url: urlparse(url).hostname.replace('www.', '') if pd.notnull(url) else 'Unknown').unique()

    # Create a color palette for domains
    color_palette = sns.color_palette("tab20", n_colors=len(all_domains))
    domain_color_map = dict(zip(all_domains, color_palette))

    for cluster_id in range(num_clusters):
        cluster_data = overview[overview['Cluster'] == cluster_id]
        merged_data = pd.merge(cluster_data, content[['Title', 'URL']], on='Title', how='left')
        merged_data['BaseDomain'] = merged_data['URL'].apply(
            lambda url: urlparse(url).hostname.replace('www.', '') if pd.notnull(url) else 'Unknown')
        domain_counts = merged_data['BaseDomain'].value_counts()

        # Create a pie chart for domain distribution in the cluster
        wedges, text, autotexts = axes[cluster_id].pie(
            domain_counts, 
            labels=None,  
            autopct='%1.1f%%',  
            startangle=90, 
            colors=[domain_color_map[domain] for domain in domain_counts.index],
            pctdistance=0.85)
        
        # Create a single legend for all clusters
        if cluster_id == 2:
            axes[cluster_id].legend(wedges, domain_counts.index, title='Domains', loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
        axes[cluster_id].set_title(f"Cluster {cluster_id}")
        
        # Print domain names for each cluster
        print(f"Cluster {cluster_id} Domains:")
        for domain, count in domain_counts.items():
            print(f"{domain}: {count} articles\n")
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()
    
    return domain_color_map
  
def analyze_date_distribution(overview, content, num_clusters):
    '''
    Analyze date distribution within each cluster.
    :param overview: DataFrame containing cluster overview.
    :param content: DataFrame containing article content.
    :param num_clusters: Number of clusters.
    '''
    num_rows = 2
    num_cols = min(num_clusters, 3)
    # Create subplots for date distribution
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(8 * num_cols, 8 * num_rows))
    axes = axes.flatten()

    cluster_colors = sns.color_palette('husl', n_colors=num_clusters)

    for cluster_id in range(num_clusters):
        row_index = cluster_id // num_cols
        col_index = cluster_id % num_cols
        # Filter data for the current cluster
        cluster_data = overview[overview['Cluster'] == cluster_id]
        merged_data = pd.merge(cluster_data, content[['Title', 'Publication Date']], on='Title', how='left')
        merged_data['Publication Date'] = pd.to_datetime(merged_data['Publication Date'])
        
        # Plot a histogram for date distribution in the cluster
        sns.histplot(data=merged_data, x='Publication Date', bins=20, kde=False, color=cluster_colors[cluster_id], ax=axes[cluster_id])
        axes[cluster_id].set_title(f"Cluster {cluster_id}")
        axes[cluster_id].tick_params(axis='x', rotation=45)

    for i in range(num_clusters, num_rows * num_cols):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.show()

def display_articles_per_cluster_grouped(overview_en, overview_de, num_clusters):
    '''
    Display the number of articles per cluster grouped by language.
    :param overview_en: DataFrame containing English cluster overview.
    :param overview_de: DataFrame containing German cluster overview.
    :param num_clusters: Number of clusters.
    '''
    # Get the count of articles per cluster for each language
    cluster_counts_en = overview_en['Cluster'].value_counts().sort_index()
    cluster_counts_de = overview_de['Cluster'].value_counts().sort_index()
    
    # Create subplots for English and German cluster counts
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    # Plot cluster counts for English
    sns.countplot(x='Cluster', data=overview_en, palette='husl', ax=axes[0])
    axes[0].set_title('Number of Articles per Cluster (English)', fontsize=12)
    axes[0].set_xlabel('Cluster ID', fontsize=12)
    axes[0].set_ylabel('Number of Articles', fontsize=12)
    
    # Annotate each bar with its count
    for i, count in enumerate(cluster_counts_en):
        axes[0].text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=14)

    # Plot cluster counts for German
    sns.countplot(x='Cluster', data=overview_de, palette='husl', ax=axes[1])
    axes[1].set_title('Number of Articles per Cluster (German)', fontsize=14)
    axes[1].set_xlabel('Cluster ID', fontsize=12)
    
    # Annotate each bar with its count
    for i, count in enumerate(cluster_counts_de):
        axes[1].text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=14)

    plt.tight_layout()
    plt.show() 
    
# ------------------------------------------------------------------------------------------   
# STEP 2: defining large functions where the helper functions are called
# Basic statistics
def get_basic_stats(en_content, de_content, nlp_english, nlp_german):
    print(en_content.describe())
    print(de_content.describe())

    # Convert the 'PublicationDate' column to datetime format
    en_content['Publication Date'] = pd.to_datetime(en_content['Publication Date'])
    de_content['Publication Date'] = pd.to_datetime(de_content['Publication Date'])

    # Extract the week information
    en_content['Week'] = en_content['Publication Date'].dt.isocalendar().week
    de_content['Week'] = de_content['Publication Date'].dt.isocalendar().week

    # Combine English and German data into a single DataFrame
    combined_weekly_counts = pd.DataFrame({
        'English': en_content['Week'].value_counts().sort_index(),
        'German': de_content['Week'].value_counts().sort_index()})

    # Fill NaN values with 0
    combined_weekly_counts = combined_weekly_counts.fillna(0)

    # Plot the graph with adjusted colors
    plt.figure(figsize=(12, 6))
    combined_weekly_counts.plot(kind='line', marker='o', color=['blue', 'orange'])
    plt.title('Weekly Publication Counts - English and German Articles')
    plt.xlabel('Weeks (August to October 2023)')
    plt.ylabel('Number of Articles')
    plt.legend(title='Language')
    plt.xticks(rotation=0)
    plt.grid(False)
    plt.show()

    #Plot publication time distribution
    en_content['Time'] = pd.to_datetime(en_content['Time'], format='%H:%M:%S%z', errors='coerce')
    de_content['Time'] = pd.to_datetime(de_content['Time'], format='%H:%M:%S%z', errors='coerce')

    # Add a new column for the hour of the day
    en_content['Hour'] = en_content['Time'].dt.hour
    de_content['Hour'] = de_content['Time'].dt.hour

    # Create bins for the time of day
    bins = [0, 6, 12, 18, 24]
    labels = ['0:00-6:00', '6:00-12:00', '12:00-18:00', '18:00-24:00']
    en_content['Time of Day'] = pd.cut(en_content['Hour'], bins=bins, labels=labels, right=False)
    de_content['Time of Day'] = pd.cut(de_content['Hour'], bins=bins, labels=labels, right=False)

    # Drop NaN values
    en_time_counts = en_content['Time of Day'].value_counts().sort_index().dropna()
    de_time_counts = de_content['Time of Day'].value_counts().sort_index().dropna()

    # Plotting pie charts for English and German
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # English pie chart
    axes[0].pie(en_time_counts, labels=en_time_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    axes[0].set_title('English Articles - Publication Time Distribution')

    # German pie chart
    axes[1].pie(de_time_counts, labels=de_time_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    axes[1].set_title('German Articles - Publication Time Distribution')

    plt.tight_layout()
    plt.show()

    #Statistics about authors (known-unknown ratio)
    authors_en = en_content["Author"]
    authors_de = de_content["Author"]

    en_frequencies = Counter(authors_en)
    de_frequencies = Counter(authors_de)
    ratio_unknown_en, ratio_known_en = calculate_author_ratios(en_frequencies)
    ratio_unknown_de, ratio_known_de = calculate_author_ratios(de_frequencies)

    print("English Article Author Ratios:")
    print(f"Ratio of Unknown Authors: {ratio_unknown_en:.2%}")
    print(f"Ratio of Known Authors: {ratio_known_en:.2%}")
    print("\nGerman Article Author Ratios:")
    print(f"Ratio of Unknown Authors: {ratio_unknown_de:.2%}")
    print(f"Ratio of Known Authors: {ratio_known_de:.2%}")

    #Frequencies of URL domains
    urls = de_content["URL"]
    domains = [urlparse(url).hostname.replace('www.', '') for url in urls]
    domain_counts = Counter(domains)
    domain_frequency_df = pd.DataFrame(list(domain_counts.items()), columns=["Domain", "Occurrences"])
    domain_frequency_df = domain_frequency_df.sort_values(by="Occurrences", ascending=False)
    print(domain_frequency_df)
    colors = sns.color_palette("husl", len(domain_frequency_df))
    plt.figure(figsize=(10, 8))
    plt.barh(domain_frequency_df['Domain'], domain_frequency_df['Occurrences'], color=colors)
    plt.title('Distribution of Articles Across Domains German')
    plt.xlabel('Number of Articles')
    plt.ylabel('Domain')
    plt.show()

    urls = en_content["URL"]
    domains = [urlparse(url).hostname.replace('www.', '') for url in urls]
    domain_counts = Counter(domains)
    domain_frequency_df = pd.DataFrame(list(domain_counts.items()), columns=["Domain", "Occurrences"])
    domain_frequency_df = domain_frequency_df.sort_values(by="Occurrences", ascending=False)
    print(domain_frequency_df)
    colors = sns.color_palette("husl", len(domain_frequency_df))
    plt.figure(figsize=(10, 8))
    plt.barh(domain_frequency_df['Domain'], domain_frequency_df['Occurrences'], color=colors)
    plt.title('Distribution of Articles Across Domains English')
    plt.xlabel('Number of Articles')
    plt.ylabel('Domain')
    plt.show()

    #Statistics about the tokens in titles
    en_statistics = calculate_title_statistics(en_content, nlp_english)
    print("Statistics of English titles in terms of tokens (rounded to full numbers):")
    print("mean, std, median, mode, variance, min, max")
    print(en_statistics)
    de_statistics = calculate_title_statistics(de_content, nlp_german)
    print("Statistics of German titles in terms of tokens (rounded to full numbers):")
    print("mean, std, median, mode, variance, min, max")
    print(de_statistics)

    #Statistics about article sentence counts
    en_content['en_num_sentences'] = en_content['Text'].apply(lambda x: get_english_sentence_count(x, nlp_english))
    de_content['de_num_sentences'] = de_content['Text'].apply(lambda x: get_german_sentence_count(x, nlp_german))
    en_mean = round(en_content['en_num_sentences'].mean())
    en_std = round(en_content['en_num_sentences'].std())
    en_median = round(en_content['en_num_sentences'].median())
    en_mode = round(en_content['en_num_sentences'].mode()[0]) 
    en_var = round(en_content['en_num_sentences'].var())
    en_min = round(en_content['en_num_sentences'].min())
    en_max = round(en_content['en_num_sentences'].max())
    de_mean = round(de_content['de_num_sentences'].mean())
    de_std = round(de_content['de_num_sentences'].std())
    de_median = round(de_content['de_num_sentences'].median())
    de_mode = round(de_content['de_num_sentences'].mode()[0])  
    de_var = round(de_content['de_num_sentences'].var())
    de_min = round(de_content['de_num_sentences'].min())
    de_max = round(de_content['de_num_sentences'].max())

    print("English Sentence Length Statistics:")
    print(f"Mean: {en_mean}, Std: {en_std}, Median: {en_median}, Mode: {en_mode}, Variance: {en_var}, Min: {en_min}, Max: {en_max}")
    print("\nGerman Sentence Length Statistics:")
    print(f"Mean: {de_mean}, Std: {de_std}, Median: {de_median}, Mode: {de_mode}, Variance: {de_var}, Min: {de_min}, Max: {de_max}")

    #Plot about article sentence counts
    plt.figure(figsize=(10, 6))
    sns.histplot(en_content['en_num_sentences'], bins=20, kde=False, color='skyblue', label='English')
    sns.histplot(de_content['de_num_sentences'], bins=20, kde=False, color='lightgreen', label='German')
    plt.title('Distribution of Sentences in English and German Articles')
    plt.xlabel('Number of Sentences')
    plt.ylabel('Article Frequency')
    plt.legend()
    plt.xticks(range(0, max(en_content['en_num_sentences'].max(), de_content['de_num_sentences'].max()) + 1, 10))
    plt.tight_layout()
    plt.show()

    #Statistics about article token counts
    en_content['en_num_tokens'] = en_content['Text'].apply(lambda x: get_english_token_count(x, nlp_english))
    de_content['de_num_tokens'] = de_content['Text'].apply(lambda x: get_german_token_count(x, nlp_german))
    en_token_mean = round(en_content['en_num_tokens'].mean())
    en_token_std = round(en_content['en_num_tokens'].std())
    en_token_median = round(en_content['en_num_tokens'].median())
    en_token_mode = round(en_content['en_num_tokens'].mode()[0])  
    en_token_var = round(en_content['en_num_tokens'].var())
    en_token_min = round(en_content['en_num_tokens'].min())
    en_token_max = round(en_content['en_num_tokens'].max())
    de_token_mean = round(de_content['de_num_tokens'].mean())
    de_token_std = round(de_content['de_num_tokens'].std())
    de_token_median = round(de_content['de_num_tokens'].median())
    de_token_mode = round(de_content['de_num_tokens'].mode()[0])
    de_token_var = round(de_content['de_num_tokens'].var())
    de_token_min = round(de_content['de_num_tokens'].min())
    de_token_max = round(de_content['de_num_tokens'].max())

    print("English Token Count Statistics:")
    print(f"Mean: {en_token_mean}, Std: {en_token_std}, Median: {en_token_median}, Mode: {en_token_mode}, Variance: {en_token_var}, Min: {en_token_min}, Max: {en_token_max}")
    print("\nGerman Token Count Statistics:")
    print(f"Mean: {de_token_mean}, Std: {de_token_std}, Median: {de_token_median}, Mode: {de_token_mode}, Variance: {de_token_var}, Min: {de_token_min}, Max: {de_token_max}")

    #Plot about article token counts
    plt.figure(figsize=(10, 6))
    sns.histplot(en_content['en_num_tokens'], bins=20, kde=False, color='skyblue', label='English')
    sns.histplot(de_content['de_num_tokens'], bins=20, kde=False, color='lightgreen', label='German')
    plt.title('Distribution of Tokens in English and German Articles')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Article Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()

    #Get token, lemma, pos-tag, NER frequencies using Stanza
    en_token_frequencies, en_lemma_frequencies, en_pos_frequencies, en_ner_frequencies = get_frequencies(en_content, 'en')
    de_token_frequencies, de_lemma_frequencies, de_pos_frequencies, de_ner_frequencies = get_frequencies(de_content, 'de')

    #NER distribution
    en_ner_cleaned = clean_and_exclude_o(en_ner_frequencies)
    de_ner_cleaned = clean_and_exclude_o(de_ner_frequencies)
    print(f"Named Entities in English Training Data:{en_ner_cleaned}")
    print(f"\nNamed Entities in German Training Data:{de_ner_cleaned}")

    #POS distribution
    print(f"Frequency of POS-tags in English articles: {en_pos_frequencies}")
    print(f"\nFrequency of POS-tags in German articles: {de_pos_frequencies}")

    #Top 10 token, lemma for each language
    print("Top 10 tokens in de_token_frequencies:")
    for token, frequency in de_token_frequencies.most_common(10):
        print(f"{token}: {frequency}")
    print("\nTop 10 lemmas in de_lemma_frequencies:")
    for lemma, frequency in de_lemma_frequencies.most_common(10):
        print(f"{lemma}: {frequency}")

    #Type-token ratios - Hypothesis 3
    en_tt_statistics = calculate_type_token_ratio(en_token_frequencies)
    print("HYPOTHESIS 3")
    print("English Type/Token Ratio:")
    print("Number of Types:", en_tt_statistics[0])
    print("Number of Tokens:", en_tt_statistics[1])
    print("Type/Token Ratio:", "%.4f" % en_tt_statistics[2])
    de_tt_statistics = calculate_type_token_ratio(de_token_frequencies)
    print("\nGerman Type/Token Ratio:")
    print("Number of Types:", de_tt_statistics[0])
    print("Number of Tokens:", de_tt_statistics[1])
    print("Type/Token Ratio:", "%.4f" % de_tt_statistics[2])
    
    # Confidence interval for TTR
    en_tt_confidence_interval = calculate_ttr_confidence_interval(en_token_frequencies, confidence=0.95)
    print(f"Confidence Interval for English Type-Token Ratio: {en_tt_confidence_interval}")
    de_tt_confidence_interval = calculate_ttr_confidence_interval(de_token_frequencies, confidence=0.95)
    print(f"Confidence Interval for German Type-Token Ratio: {de_tt_confidence_interval}")

    # Randomization test for TTR
    p_value_ttr = randomization_test_ttr(en_token_frequencies, de_token_frequencies)
    print(f"P-Value for Difference in Type-Token Ratios between English and German: {p_value_ttr}")

# -------------------------------------------------------------------------------------   
# NERC for extracting GEOGRAPHICAL LOCATIONS with SpaCy
def visualise_most_freq_locations(data, language, nlp_model, special_cases=None):
    '''
    Visualise the top 10 geographical locations mentioned in the given dataset.
    :param data: DataFrame containint 'Text' column with the text data.
    :param language (str): define the language of the dataset
    :param nlp_model: the Spacy language model used for NER
    :param special_cases: dict, optional to specify rules for NER
    :return None
    '''
    all_unique_location_names = []
    # Apply the extract_countries function to each row and create a new column 'Location_Names_Set'
    data['Location_Names_Set'] = data.apply(lambda row: extract_countries(row['Text'], nlp_model, special_cases), axis=1)
    # Extend the list of all unique location names with the sets from each row
    all_unique_location_names.extend(data['Location_Names_Set'])

    # Count the occurrences of each location in the entire dataset
    total_location_counts = Counter(item for sublist in all_unique_location_names for item in sublist)
    
    # Get the top 10 most common locations
    top_10_countries = total_location_counts.most_common(10)
    print("HYPOTHESIS 1")
    print(f"Top 10 Geograhical Locations Mentioned in {language} Articles:")
    print(top_10_countries) # Hypothesis 1
    
    colors = sns.color_palette("husl", len(top_10_countries))

    def plot_location_counts(location_counts, language, colors):
        labels, counts = zip(*location_counts)
        plt.bar(labels, counts, color=colors)
        plt.title(f"Top 10 Mentioned Geographical Locations in {language.capitalize()} Articles")
        plt.xlabel("Location")
        plt.ylabel("Article Frequency")
        plt.xticks(rotation=45, ha="right")
        plt.show()

    #Plot top 10 mentioned countries in German articles with different colors
    plot_location_counts(top_10_countries, language, colors)

def visualize_locations_across_domains(data, language, nlp_model, special_cases=None):
    '''
    Visualise the distribution of location NEs across different domains.
    :param data: DataFrame containint 'Text' column with the text data.
    :param language (str): define the language of the dataset
    :param nlp_model: the Spacy language model used for NER
    :param special_cases: dict, optional to specify rules for NER
    :return None
    '''
    # Apply the function to the 'Text' and 'URL' columns
    result_df = data.apply(lambda row: extract_countries_and_domains(row['Text'], row['URL'], nlp_model, special_cases), axis=1)

    # Create a new DataFrame with columns 'Domain' and 'Location_Names_Set'
    data[['Domain', 'Location_Names_Set']] = pd.DataFrame(result_df.tolist(), index=result_df.index)

    # Combine sets from all articles into a list
    all_domain_location_names = []
    all_domain_location_names.extend(data[['Domain', 'Location_Names_Set']].values)

    # Create a Counter from the combined sets
    total_domain_location_counts = Counter()

    # Display the top domains and their associated counts
    for domain, location_names_set in all_domain_location_names:
        total_domain_location_counts[domain] += len(location_names_set)

    # Display all domains and their associated counts
    all_domains_count = total_domain_location_counts.most_common()

    # Create a pie chart
    labels, counts = zip(*all_domains_count)
    colors_domains = sns.color_palette("husl", len(all_domains_count))

    plt.figure(figsize=(10, 10))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors_domains, startangle=140, pctdistance=0.85)
    plt.title(f"Distribution of Location Named Entities for All Web Sources in {language} Articles")
    plt.show()

# -------------------------------------------------------------------------------------   
#CLUSTERING
def get_clusters(data, nlp, num_clusters, word_vector_path, lang):
    '''
    Get clusters, overview, keywords, and document representations for a given set of articles.
    :param data: DataFrame containing articles data.
    :param nlp: SpaCy NLP pipeline.
    :param num_clusters: Number of clusters.
    :param word_vector_path: Path to the word vector model.
    :param lang: Language of the articles ('english' or 'german').
    :return: Tuple containing clusters, overview, keywords, and document representations.
    '''
    articles = data["Text"]
    if lang == 'English':
        stop_words = stopwords.words('english')
    else:
        stop_words = stopwords.words('german')
    
    # Get TF-IDF representation and keywords for articles
    doc_representations, keywords = get_tfidf_representation(articles, 
                                                             lambda x: get_lemmas(x, stop_words, nlp),
                                                             word_vector_path)

    # Cluster articles based on document representations
    clusters = cluster_articles(doc_representations, num_clusters)
    overview = pd.DataFrame({'Author': data["Author"], 'Title': data["Title"], 'Cluster': clusters})

    # Calculate keyword frequencies and get the 10 most frequent ones out
    keyword_freq = Counter(keyword for cluster_keywords in keywords for keyword in cluster_keywords)
    print(f"Top Keywords: {keyword_freq.most_common(10)}")
    
    return clusters, overview, keywords, doc_representations

def calculate_silhouette_scores(doc_representations, cluster_range):
    '''
    Calculate silhouette scores for a given range of cluster numbers.
    :param doc_representations: List of document representations.
    :param cluster_range: Range of cluster numbers to evaluate.
    :return: List of silhouette scores.
    '''
    silhouette_scores = []
    # Calculate silhouette scores for different cluster numbers
    for num_clusters in cluster_range:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(doc_representations)
        silhouette_scores.append(silhouette_score(doc_representations, cluster_labels))

    return silhouette_scores

def plot_combined_silhouette_scores(cluster_range, silhouette_scores_en, silhouette_scores_de):
    '''
    Plot silhouette scores for English and German clusters.
    :param cluster_range: Range of cluster numbers.
    :param silhouette_scores_en: Silhouette scores for English clusters.
    :param silhouette_scores_de: Silhouette scores for German clusters.
    '''
    plot_silhouette_scores(cluster_range, silhouette_scores_en, 'English', 'blue')
    plot_silhouette_scores(cluster_range, silhouette_scores_de, 'German', 'orange')

    plt.title('Silhouette Score for Optimal Cluster Number')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.legend()
    plt.show()
    
def measure_balance(clusters_overview): #Hypothesis 2
    '''
    Measure the balance of articles distribution across clusters.
    :param clusters_overview: DataFrame containing cluster overview.
    :return: Standard deviation of cluster counts for the specific language.
    '''
    # Count the number of articles in each cluster
    cluster_counts = clusters_overview['Cluster'].value_counts().sort_index()
    # Calculate the standard deviation of cluster counts
    std_dev = cluster_counts.std()
    return std_dev

# ------------------------------------------------------------------------------------- 
# STEP 3: MAIN
def main():
    # Check if the number of command-line arguments is correct
    if len(sys.argv) != 2:
        print("Error in usage: python run_all_analysis.py OUT_DIR")
        return
    
    OUT_DIR = sys.argv[1]  # Get the OUT_DIR directory from the command-line argument
    
    # Ask user to define if analyses should be run on train/test data
    functionality = input("Do you want to run analyses on train or test data? Enter 'train'/'test': ") 
    
    # Load files
    if functionality.lower() == 'train':
        en_tsv_file = os.path.join(OUT_DIR, "eng", "train", "immigration_immigrant_filtered.tsv")
        de_tsv_file = os.path.join(OUT_DIR, "deu", "train", "Einwanderung_Einwanderer_Flüchtlinge_Migration_filtered.tsv")
    elif functionality.lower() == 'test':
        en_tsv_file = os.path.join(OUT_DIR, "eng", "test", "immigration_immigrant_filtered.tsv")
        de_tsv_file = os.path.join(OUT_DIR, "deu", "test", "Einwanderung_Einwanderer_Flüchtlinge_Migration_filtered.tsv")
    else:
        print("Error in usage. Please enter 'train' or 'test'.")
        return

    # Read in with pandas
    en_content = pd.read_csv(en_tsv_file, sep="\t", header=0)
    de_content = pd.read_csv(de_tsv_file, sep="\t", header=0)
    
    # Define word_vector_paths
    word_vector_path_en = os.path.join(OUT_DIR, "data", "wiki-news-300d-1M.vec")
    word_vector_path_de = os.path.join(OUT_DIR, "data", "wiki.de.vec")
    
    # ----------------------------------------------------------------
    # Preprocessing
    en_content = preprocess_dataframe(en_content)
    de_content = preprocess_dataframe(de_content)
    
    # Creating stanza pipelines
    nlp_english = stanza.Pipeline('en')
    nlp_german = stanza.Pipeline('de')
    # Creating spacy pipelines
    nlp_english_spacy = spacy.load("en_core_web_sm")
    nlp_german_spacy = spacy.load("de_core_news_md")
    
    # ----------------------------------------------------------------
    # Basic analysis for both languages
    get_basic_stats(en_content, de_content, nlp_english, nlp_german)
    
     #----------------------------------------------------------------
     # NERC information about geographical locations
    visualise_most_freq_locations(de_content, "German", nlp_german_spacy, special_cases={"deutschland": "Deutschland", "deutschlands": "Deutschland", "italien": "Italien", "italiens": "Italien"})
    visualize_locations_across_domains(de_content, "German", nlp_german_spacy, special_cases={"deutschland": "Deutschland", "deutschlands": "Deutschland", "italien": "Italien", "italiens": "Italien"})
    visualise_most_freq_locations(en_content, "English", nlp_english_spacy, special_cases={"us": "U.S.", "u.s.": "U.S.", "the united states": "U.S.", "united states": "U.S."})
    visualize_locations_across_domains(en_content, "English", nlp_english_spacy, special_cases={"us": "U.S.", "u.s.": "U.S.", "the united states": "U.S.", "united states": "U.S."})

    #----------------------------------------------------------------
    # CLUSTERS (EN)
    num_clusters = 3
    clusters_en, overview_en, keywords_en, doc_representations_en = get_clusters(en_content, nlp_english, num_clusters, word_vector_path_en, 'English')
    # CLUSTERS (DE)
    clusters_de, overview_de, keywords_de, doc_representations_de = get_clusters(de_content, nlp_german, num_clusters, word_vector_path_de, 'German')
    
    # Silhouette Scores
    cluster_range = range(2, 11)
    silhouette_scores_en = calculate_silhouette_scores(doc_representations_en, cluster_range)
    silhouette_scores_de = calculate_silhouette_scores(doc_representations_de, cluster_range)
    # Plot silhouette scores for both languages
    plot_combined_silhouette_scores(cluster_range, silhouette_scores_en, silhouette_scores_de)
    
    # Extract top keywords per cluster
    print_cluster_top_keywords(keywords_en, clusters_en, num_clusters)
    print_cluster_top_keywords(keywords_de, clusters_de, num_clusters)

    # Plot distribution of articles per cluster
    display_articles_per_cluster_grouped(overview_en, overview_de, num_clusters)
    
    # Measure cluster imbalance across English and German clusters
    balance_en = measure_balance(overview_en)
    balance_de = measure_balance(overview_de)
    print("HYPOTHESIS 2")
    print(f"Balance in the English data across clusters using standard deviation: {balance_en:.2f}")
    print(f"Balance in the German data across clusters using standard deviation: {balance_de:.2f}")

    # Generate cluster wordclouds
    generate_cluster_wordcloud(keywords_en, clusters_en, num_clusters)
    generate_cluster_wordcloud(keywords_de, clusters_de, num_clusters)

    # Generate plots to explore connection between clusters and domains
    domain_color_map_en = analyze_domain_distribution(overview_en, en_content, num_clusters)
    domain_color_map_de = analyze_domain_distribution(overview_de, de_content, num_clusters)

    # Generate plots to explore connection between clusters and publication date
    analyze_date_distribution(overview_en, en_content, num_clusters)
    analyze_date_distribution(overview_de, de_content, num_clusters)

if __name__ == "__main__":
    main()