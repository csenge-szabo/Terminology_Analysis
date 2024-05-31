# Language as Data: Terminology Analysis of News Articles about Immigration

## Author
**Name:** Csenge Szabó  

## Research Purpose
The purpose of this research project is to examine the topic of immigration in online news articles between the period August and October 2023. I aim to compare the news media coverage of this topic in articles extracted in two languages. My initial hypothesis is that news written in German published on German websites are likely to discuss immigration related to different geographical regions (such as Italy) than news articles written in English, posted on websites in US (such as Mexico). My second hypothesis is that articles could be clustered based on their unique keywords, which might allow us to discover connections between metadata (such as web sources, publication date) and thematic clusters.

## Languages Analyzed
- English (eng)
- German (deu)

## Selected News Domains
- **US:** cnn.com, chicago.suntimes.com, boston.com, dailynews.com, denverpost.com, forbes.com, foxnews.com, latimes.com, mercurynews.com, twincities.com, upi.com, chelsearecord.com, azcapitoltimes.com, dailycaller.com, journalrecord.com, nypost.com, sentinelcolorado.com, dailybreeze.com, dailybulletin.com, dailydemocrat.com, lowellsun.com, minnpost.com, reviewjournal.com, sandiegouniontribune.com, thevindicator.com, thetwincities.com, dailycaller.com, bostonherald.com, thedailyrecord.com, deseret.com
- **Germany:** zeit.de, spiegel.de, berliner-zeitung.de, zdf.de, sueddeutsche.de, welt.de, dw.com, tagesschau.de

## Directory Structure
- `util_html.py`: Helper functions for document retrieval.
- `util_analysis.py`: Helper functions for data analysis.
- `get_all_documents.py`: Script for crawling news from MediaStack.
- `run_all_analyses.py`: Script for data analysis.
- Ensure all scripts are in the same directory.

## Recommended Usage
1. **Pre-trained FastText Models:**
   - Download `wiki-news-300d-1M.vec` from [FastText English Vectors](https://fasttext.cc/docs/en/english-vectors.html).
   - Download `wiki.de.vec` from [FastText Pretrained Vectors](https://fasttext.cc/docs/en/pretrained-vectors.html).
   - Place them into the `OUT_DIR/data` subdirectory.
2. **Install Required Python Modules** (You can use pip install -r requirements.txt)
3. **Run run_all_analyses.py OUT_DIR in the command line.** OUT_DIR is an argument which you must use to specify a file path where the subdirectories ('eng', 'deu', 'data') exist on your disk. You will be prompted to enter 'train' or 'test' to indicate if you wish to run analyses on the train data or test data. To answer the hypotheses, you need to run the script on the test data provided.

## Code Details

### Crawling Procedure (`get_all_documents.py`)
- **Prerequisites**: Obtain a Media Stack API key at [Media Stack Dashboard](https://mediastack.com/dashboard).
- **Usage**: `python get_all_documents.py OUT_DIR`
- **Steps**:
  1. **API Key**: Prompt user for Media Stack API key.
  2. **Directory Setup**: Create `deu` and `eng` directories with subdirectories ('train' and 'test') in `OUT_DIR`.
  3. **Crawl Parameters**: Set parameters for crawling using the `languages` variable.
  4. **Collect News**: Execute `collect_news_media_stack` to retrieve news articles.
  5. **Filter Articles**: Ensure only 200 unique articles per language are retained using `filter_articles`.
  6. **Split Data**: Divide articles into training (80%) and test (20%) using `split_train_test`.
  7. **Clean Up**: Remove unwanted files, retaining only train or test data with `clean_folder`.

### Analysis Procedure (`run_all_analyses.py`)
- **Usage**: `python run_all_analyses.py OUT_DIR`
- **Steps**:
  1. **Data Selection**: Choose between 'train' or 'test' data.
  2. **Load Data**: Import TSV files, set paths to FastText models.
  3. **Preprocessing**: Utilize Stanza and SpaCy for content analysis.
  4. **Basic Statistics**: Obtain and plot data insights with `get_basic_stats`.
  5. **Named Entity Analysis**: Examine NE distribution and domain correlations.
  6. **Clustering Analysis**:
     - Perform KMeans clustering with TF-IDF and FastText.
     - Analyze and plot cluster details including keywords, imbalance, and distributions.

## Important Remarks
- **Execution Time**: Expect 30-60 minutes per script. Plan your time accordingly.
- **Dataset Variability**: Content diversity may vary due to dynamic changes at Media Stack. Note the crawling date: October 7, 2023.
- **Selective Execution**: For targeted analysis, comment out irrelevant sections in the `main` function of `run_all_analyses.py`.

## Licensing and Usage
- **Media Stack API**:
  - **Documentation**: [Media Stack Documentation](https://mediastack.com/documentation)
  - **License**: [Media Stack Terms of Use](https://mediastack.com/terms)
  - **Scope**: Granted for non-commercial, personal research use. Do not redistribute data outside your application.

### Scope of License
- **Grant of License**: apilayer grants a non-transferable, non-exclusive, revocable license to access the mediastack API and use its services.
- **Usage Restrictions**:
  - API data is for personal use by end-users of your application.
  - Storing, distributing, or exploiting API data beyond personal use is not permitted.
  - API data must remain within your application; external distribution is prohibited.
