import os
import http.client
import urllib.parse
import json
import pandas as pd
from util_html import *
from datasets import Dataset
import sys

#Step 1: Defining functions
def extract_metadata(article):
    """
    Take an article and extract information about it. It is called inside the function 'collect_news_media_stack'.
    :param article 
    return: date, time, title, url, category, country
    """
    published_at = article['published_at']
    if published_at:
        date, time = published_at.split("T")        
    else:
        date = ""
        time = ""

    # Extract meta data
    url = article ['url']
    title= article['title'] 
    
    # category associated with the given news article
    category = article['category']
    
    # country code associated with given article 
    country = article ['country']
    
    return date, time, title, url, category, country

def collect_news_media_stack(OUT_DIR, access_key, lang, keywords, countries, mode="w", offset=None):
    """
    Crawls news articles from Media Stack using API search and gathers articles into a TSV file.
    :param OUT_DIR(str): the file path where the TSV file should to be saved to
    :param access_key(str): your API access key
    :param lang(str): ISO language code for the search ("de" for German or "en" for English)
    :param keywords(str): string of keywords for searching the news articles
    :param countries(str): countries of your search
    :param mode(str): accepts 'w' for writing or 'a' for appending the TSV file, defaults to 'w'
    :param offset(int): to specify a pagination offset value for your API search
    :return None
    """
    contents_set = set()

    conn = http.client.HTTPConnection('api.mediastack.com')

    params = urllib.parse.urlencode({
        'access_key': access_key,
        'keywords': keywords,
        'sort': 'published_asc',
        'languages': lang,
        'countries': countries,
        'date': '2023-08-01,2023-11-01',
        'limit': 100,
        'sources': '-yahoo', #the yahoo site caused many errors, so it was excluded from the search
        'offset': offset})

    conn.request('GET', '/v1/news?{}'.format(params))

    res = conn.getresponse()
    data = res.read()

    query_content = data.decode('utf-8')
    query = json.loads(query_content)
    
    if lang == "de":
        output_dir_lang = "deu"
    elif lang == "en":
        output_dir_lang = "eng"
    else:
        print("param for 'lang' is invalid, please use 'en' or 'de'")

    outfile = os.path.join(OUT_DIR, output_dir_lang, keywords.replace(' ', '_') + "_full_dataset.tsv")

    with open(outfile, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("Publication Date\tTime\tAuthor\tTitle\tURL\tText\n") # Write a header in TSV file if opened in mode 'w'

        articles = query.get("data", [])
        for article in articles:
            date, time, title, article_url, category, country = extract_metadata(article)

            article_content = url_to_html(article_url)
            # Check if HTML content is None
            if article_content is None:
                print(f"HTML content is None. Skipping URL: {article_url}")
                continue
            
            author = parse_author(article_content)
            content = parse_news_text(article_content)

            content = content.replace("\n", "")
            if content == '': # Skip article if the content is empty
                continue

            if content in contents_set: # Skip article if the same content was already gathered in the same round
                continue 

            contents_set.add(content)

            # Separate fields by tabulators (\t) for TSV file
            output = "\t".join([date, time, author, title, article_url, content])
            f.write(output + "\n")
    
    print("The news articles are being crawled, please wait.")
    
def filter_articles(input_filepath, language):
    """
    Takes an input TSV file, filters 200 unique lines based on content and URL, and creates a new file in the same folder as the input_filepath.
    :param input_filepath (str): Path to the input TSV file.
    :param language (str): ISO language code ("de" for German or "en" for English).
    :return None
    """
    output_filepath = input_filepath.replace('_full_dataset.tsv', '_filtered.tsv')
    unique_contents = set()
    unique_lines = []
    max_lines = 200  # filter to max 200 articles

    # Defining a list of acceptable URLs for each language to filter out news articles outside of Germany and outside of US (although we set the countries parameter when calling collect_news_media_stack, some articles from other countries might still be present as I experienced)
    if language == "de":
        allowed_urls = [".zeit.de","spiegel.de","berliner-zeitung","zdf","sueddeutsche",".welt.de","dw.com","tagesschau"]
    elif language == "en":
        allowed_urls = ["cnn","chicago.suntimes","boston.com","dailynews.com","denverpost","forbes.com","foxnews.com","latimes.com","mercurynews.com","twincities.com","upi.com","chelsearecord", "azcapitoltimes","dailycaller","journalrecord","nypost","sentinelcolorado","dailybreeze","dailybulletin","dailydemocrat","lowellsun","minnpost","reviewjournal","sandiegouniontribune","thevindicator","thetwincities", "dailycaller", "bostonherald","thedailyrecord", "deseret"]

    # Opening and reading in the input file with readlines
    with open(input_filepath, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines() 
    
    # Iterating through each line containing information about news articles
    for line in lines:
        parts = line.split('\t')
        url = parts[4]
        text = parts[5]

        if len(parts) > 6:  # Excluding lines with incorrect segmentation due to multiple tabs
            continue
            
        if len(text) < 100: # Excluding articles with fewer than 100 characters
        	continue

        # Check if any of the allowed URLs is present in the article's URL
        is_allowed_url = any(allowed_url in url for allowed_url in allowed_urls)

        # Check if the first 100 characters of the text content is unique to avoid duplicate content
        if is_allowed_url:
            beginning_content = parts[5][:100] 
            if beginning_content not in unique_contents:
                unique_contents.add(beginning_content)
                unique_lines.append(line)

        if len(unique_lines) >= max_lines:  # Check if the maximum limit of 200 articles is reached, if so break the loop
            break

    # Write the filtered lines to the output file (200 lines)
    with open(output_filepath, 'w', encoding='utf-8') as output_file:
        output_file.write("Publication Date\tTime\tAuthor\tTitle\tURL\tText\n") # Write a header
        output_file.writelines(unique_lines) 
    
    print("The news articles have been filtered.")
    
def split_train_test(input_filepath, seed, test_size=0.2):
    """
    Split a TSV file randomly into train and test datasets and save them in the same directory as the input file.
    :param input_filepath (str): The path to the input TSV file.
    :seed (int): seed for reproducibility
    :test_size(float): The proportion of the dataset to include in the test split. Defaults to 0.2.
    :return None
    """
    directory, filename = os.path.split(input_filepath)
    
    # Defining output filepaths
    train_output_filepath = os.path.join(directory, "train", filename)
    test_output_filepath = os.path.join(directory, "test", filename)

    pandas_dataset = pd.read_csv(input_filepath, sep="\t", on_bad_lines="warn")
    dataset = Dataset.from_pandas(pandas_dataset)

    # Split the dataset into train and test set
    splitted = dataset.train_test_split(test_size=test_size)
    train_dataset = splitted["train"]
    test_dataset = splitted["test"]

    with open(train_output_filepath, 'w', encoding='utf-8') as train_output_file:
        train_dataset.to_pandas().to_csv(train_output_file, sep="\t", index=False)

    with open(test_output_filepath, 'w', encoding='utf-8') as test_output_file:
        test_dataset.to_pandas().to_csv(test_output_file, sep="\t", index=False)
    
    print("The train and test files have been placed in 'train' and 'test' subdirectories.")
    
def clean_folder(folder_path):
    """
    Takes a folder path as input and removes every file that does not contain the strings 'train' or 'test'.
    :param folder_path (str): file path to folder to be cleaned
    :return None
    """
    for filename in os.listdir(folder_path):
        if "train" in filename or "test" in filename:
            continue
        else:
            removable_path = os.path.join(folder_path, filename)
            os.remove(removable_path)
            print(f"The following file was removed: {removable_path}")
            
#Step 2: Create a main function in which the functions above are called
def main():
    # Check if the number of command-line arguments is correct
    if len(sys.argv) != 2:
        print("Error in usage: python get_all_documents.py OUT_DIR")
        return
    
    # Ask user to define the API access key
    access_key = input("Enter your API access key for Media Stack: ") 
    
    OUT_DIR = sys.argv[1]  # Get the output directory from the command-line argument

    # Create directories for the two languages and their subdirectories
    os.makedirs(os.path.join(OUT_DIR, "deu", "test"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "deu", "train"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "eng", "test"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "eng", "train"), exist_ok=True)
    
    # Define languages and their respective parameters
    languages = {
        "de": {"keywords": "Einwanderung Einwanderer Flüchtlinge Migration", "countries": "de"},
        "en": {"keywords": "immigration immigrant","countries": "us"}}

    # Calling collect_news_media_stack
    # Loop through each language and corresponding parameters
    for lang, params in languages.items():
        language = lang
        keywords = params["keywords"]
        countries = params["countries"]

        # First call the function for each language with mode "w"
        collect_news_media_stack(OUT_DIR, access_key, language, keywords, countries, mode="w")

        # Call the function again in a range for each language with mode "a" to crawl more entries
        # This is needed because Media Stack only allows us to crawl 100 entries at a time
        for i in range(1, 18):
            offset = i * 100  # Calculate the offset based on the loop index
            collect_news_media_stack(OUT_DIR, access_key, language, keywords, countries, mode="a", offset=offset)

    print("All news articles have been crawled.")

    # Define directory paths within OUT_DIR
    deu_dir = os.path.join(OUT_DIR, "deu")
    eng_dir = os.path.join(OUT_DIR, "eng")
    
    # Modify keywords to replace spaces with underscores because file names should have no white spaces
    de_keywords = languages["de"]["keywords"].replace(" ", "_")
    en_keywords = languages["en"]["keywords"].replace(" ", "_")

    # Calling filter_articles function
    input_tsv_file_de = os.path.join(deu_dir, f'{de_keywords}_full_dataset.tsv')
    input_tsv_file_en = os.path.join(eng_dir, f'{en_keywords}_full_dataset.tsv')

    filter_articles(input_tsv_file_de, language="de")
    filter_articles(input_tsv_file_en, language="en")

    # Calling split_train_test function
    input_filtered_filepath_de = os.path.join(deu_dir, f'{de_keywords}_filtered.tsv')
    input_filtered_filepath_en = os.path.join(eng_dir, f'{en_keywords}_filtered.tsv')

    split_train_test(input_filtered_filepath_de, 5)
    split_train_test(input_filtered_filepath_en, 5)

    # Cleaning up any files from the directories that are not the train or test data sets
    clean_folder(deu_dir)
    clean_folder(eng_dir)

#Step 3: call the main function
if __name__ == "__main__":
    main() # I read about this in the Python documentation https://docs.python.org/3/library/__main__.html, until now I was not familiar with using this.