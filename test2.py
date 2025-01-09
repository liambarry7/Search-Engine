import pandas
import regex

from bs4 import BeautifulSoup, Comment

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def webScraper():
    kgrams = []

    # get the name of html file from csv file
    videogame_details = pandas.read_csv('videogame-labels.csv')
    videogame_dict = videogame_details.to_dict(orient='records') # Convert CVS file from pandas data structure to dictionary

    for record in videogame_dict:
        print(record)

    # get the url from each record reading to scrape, add to list
    list_of_urls = []
    for record in videogame_dict:
        url = record['url']
        list_of_urls.append(url)

    # print(list_of_urls)

    # Dictionary to store all tokens from one document
    list_of_document_tokens = []

    # beautiful soup
    for url in list_of_urls:
        # remove unneccessary parts of url for html file search
        pattern = regex.compile(r"^.*?/.*?/") # match from start, any character until first /, then repeat until next one
        simpleURL = regex.sub(pattern, "", url)
        # print(simpleURL)

        # r = requests.get('videogames/' + simpleURL)
        r = open('videogames/' + simpleURL, 'r')
        soup = BeautifulSoup(r.read(), 'html5lib')
        soup.prettify()
        # print(soup)


        # --------------------------------------------
        # Get game title for kgram/raw name
        raw_title = soup.find_all("title")
        game_title = regex.sub(r"GameSpy: ", "", raw_title[0].get_text())
        kgrams.append(game_title)
        # print(game_title)
        # --------------------------------------------

        # Remove css and js from scraped html
        for data in soup(['style', 'script']):
            data.decompose()

        # Get all text from under the content id
        data = soup.find_all(id='content')
        refined_data = str(data[0].get_text())
        pattern = regex.compile(r"\s+")
        simplified_data = regex.sub(pattern, " ", refined_data)
        # print(simplified_data)

        # --- remove punctuation and lowercase ---
        removed_punctuation = remove_punc(simplified_data)

        # --- Tokenise ---
        tokens = tokeniser(removed_punctuation)

        # --- Remove stopwords ---
        tokened_removed_stopwords = remove_stopwords(tokens)

        # --- Lemmatize ---
        lemmatized_words = lemmatizer(tokened_removed_stopwords)

        vg_tokens = {"Name": game_title, "Tokens": lemmatized_words}
        print(vg_tokens)
        list_of_document_tokens.append(vg_tokens)

    for w in list_of_document_tokens:
        print(w)



def getQuery():
    query = input("Please enter your query: ")
    return query

def csv_reader():
    # Get the name of html file from csv file
    videogame_details = pandas.read_csv('videogame-labels.csv')
    videogame_dict = videogame_details.to_dict(orient='records')  # Convert CVS file from pandas data structure to dictionary

    for record in videogame_dict:
        print(record)

    # return videogame_dict

def web_scrapper(soup):
    # soup = beautifulSoup()
    return 0

def remove_punc(text_data):
    # Removing punctuation from text
    removed_punctation = regex.sub(r"[^\w\s]", "", text_data)
    removed_punctation = removed_punctation.lower()
    # print(removed_punctation)
    return removed_punctation

def tokeniser(text_data):
    # Tokenise words
    tokens = word_tokenize(text_data)
    return tokens

def remove_stopwords(text_data):
    stopwords_list = set(stopwords.words('English'))
    tokenised_words = []
    for word in text_data:
        if word not in stopwords_list:
            tokenised_words.append(word)

    return tokenised_words

def lemmatizer(tokens):
    lm = WordNetLemmatizer()
    lemmatized_words = []

    for word in tokens:
        lemmatized_words.append(lm.lemmatize(word))

    return lemmatized_words

def stemming(tokens):
    print("Stemming")

def main():
    # query = getQuery()
    # print("Your query is:", query)
    # webScraper()
    csv_reader()

main()