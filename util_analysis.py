import stanza
import string
import numpy as np
import matplotlib.pyplot as plt


def tokenize_title(title, nlp_pipeline):
    '''
    Tokenize a title using the specified Stanza NLP pipeline.
    :param title: title to be tokenized.
    :param nlp_pipeline:  Stanza NLP pipeline.
    :return: List of tokenized words.
    '''
    doc = nlp_pipeline(title)
    tokens = [word.text for sent in doc.sentences for word in sent.words]
    return tokens

def get_english_sentence_count(text, nlp_english):
    '''
    Get the count of sentences in English text using Stanza NLP.
    :param text: input text.
    :return: count of sentences.
    '''
    doc = nlp_english(text)
    return len(doc.sentences)

def get_german_sentence_count(text, nlp_german):
    '''
    Get the count of sentences in German text using Stanza NLP.
    :param text: input text.
    :return: count of sentences.
    '''
    doc = nlp_german(text)
    return len(doc.sentences)

def get_english_token_count(text, nlp_english):
    '''
    Get the count of tokens in English text excluding punctuation.
    :param text: input text.
    :return: count of tokens.
    '''
    doc = nlp_english(text)
    tokens = [token.text for sentence in doc.sentences for token in sentence.tokens if token.text not in string.punctuation]
    return len(tokens)

def get_german_token_count(text, nlp_german):
    '''
    Get the count of tokens in German text excluding punctuation.
    :param text: input text.
    :return: count of tokens.
    '''
    doc = nlp_german(text)
    tokens = [token.text for sentence in doc.sentences for token in sentence.tokens if token.text not in string.punctuation]
    return len(tokens)

def clean_ner_tags(tags):
    '''
    Clean Named Entity Recognition (NER) tags.
    :param tags: List of NER tags.
    :return: List of cleaned NER tags.
    '''
    cleaned_tags = [tag.replace("B-", "").replace("I-", "").replace("E-", "").replace("S-", "") for tag in tags]
    return cleaned_tags

def clean_and_exclude_o(ner_dict):
    '''
    Clean and exclude 'O' (non-NER) tags from a NER dictionary.
    :param ner_dict: Dictionary of NER tags and their counts.
    :return: Dictionary with 'O' tags excluded and cleaned NER tags.
    '''
    cleaned_dict = {}
    for key, value in ner_dict.items():
        if key != 'O':
            key_cleaned = clean_ner_tags([key])[0]
            cleaned_dict[key_cleaned] = cleaned_dict.get(key_cleaned, 0) + value
    return cleaned_dict

def get_top_tfidf_features(row, terms, top_n=25):
    '''
    Get the top TF-IDF features from a row.
    :param row: Array representing TF-IDF values.
    :param terms: List of terms corresponding to TF-IDF values.
    :param top_n: Number of top features to retrieve.
    :return: List of top TF-IDF features, array of indices corresponding to the top features.
    '''
    top_ids = np.argsort(row)[::-1][:top_n]
    top_features = [terms[i] for i in top_ids]
    return top_features, top_ids

def plot_silhouette_scores(cluster_range, silhouette_scores, label, color):
    '''
    Plot silhouette scores for a cluster range to find out about ideal cluster numbers.
    '''
    plt.plot(cluster_range, silhouette_scores, marker='o', label=label, color=color)
