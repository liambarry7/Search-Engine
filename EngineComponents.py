import math

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

    # count = 0
    # for record in videogame_dict:
    #     print(record)
    #     count += 1
    # print(count)

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
        list_of_document_tokens.append(vg_tokens)


    # ---- testing ----
    # for w in list_of_document_tokens:
    #     print(w)

    # testTokens = list_of_document_tokens[0]["Tokens"]
    # print (testTokens)
    # x = term_frequency("hackgu", testTokens)
    # print ("tf ", x)
    #
    # print(inverse_document_frequency("wwe", list_of_document_tokens))

    # tfidf_list = tf_idf("ICO", list_of_document_tokens)
    # for i in tfidf_list:
    #     print(i)

    multi = multi_term_tdidf("wwe raw soldier", list_of_document_tokens)
    for i in multi:
        print(i)

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

def web_scrapper():
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

def term_frequency(term, doc_tokens):
    # if no term in doc, score = 0
    # tf = 1 + math.log(tf)

    frequency = 0
    for i in doc_tokens:
        if term == i:
            frequency += 1

    if (frequency > 0):
        tf = 1 + math.log(frequency, 10) # log x, base 10
        return tf

    else:
        return 0

def inverse_document_frequency(term, vg_list):
    # idf = math.log(n / df)
    # n = collection size
    # df = no of docs term is in
    # vg list = dictionary of each game's tokens

    # print("\n\n")

    # Basic df check
    document_frequency = 0
    for vg in vg_list:
        for token in vg["Tokens"]:
            if token == term:
                document_frequency += 1
                break # Break inner loop as term already found once in doc
        else:
            continue # Continue until term occurs or not found

    collection_size = len(vg_list)

    # Handle if term does not appear in any document
    if document_frequency == 0:
        return 0
    else:
        idf = math.log((collection_size / document_frequency), 10)
        return idf

def tf_idf(term, vg_list):
    # tf * idf
    print("\n")

    # idf = inverse_document_frequency(term, vg_list)
    # print(idf)
    vg_tfidf = []

    for i in range(len(vg_list)):
        # term_dict = {"doc name": vg_list[i]["Name"], "tf": term_frequency(term, vg_list[i]["Tokens"])}
        # print(term_dict)
        tfidf = term_frequency(term, vg_list[i]["Tokens"]) * inverse_document_frequency(term, vg_list)
        # print(tfidf)

        doc_tfidf = {"Doc name": vg_list[i]["Name"], "tf-idf": tfidf}
        vg_tfidf.append(doc_tfidf)

    # returns list of each doc's tf-idf of the term
    return vg_tfidf

def multi_term_tdidf(query, vg_list):
    query_terms = query.split(" ")
    print(query_terms)

    docs_tfidf = []

    for i in range(len(vg_list)):

        term_tdidf_list = []

        for j in range(len(query_terms)):
            tfidf = term_frequency(query_terms[j], vg_list[i]["Tokens"]) * inverse_document_frequency(query_terms[j], vg_list)
            # print(tfidf)

            t = {"Term": query_terms[j], "tf-idf": tfidf}
            term_tdidf_list.append(t)

        all_terms_tfidf = {"Doc name": vg_list[i]["Name"], "Query tf-idf": term_tdidf_list}
        docs_tfidf.append(all_terms_tfidf)

    return docs_tfidf

def main():
    # query = getQuery()
    # print("Your query is:", query)
    webScraper()
    # csv_reader()

    list = ['hackgu', 'vol', '1', 'rebirth', 'ps2', 'year', 'event', 'first', 'hack', 'installment', 'popular', 'dangerous', 'online', 'game', 'world', 'resurrected', 'haseo', 'terror', 'death', 'player', 'thrown', 'quest', 'revenge', 'search', 'mysterious', 'pk', 'player', 'killer', 'triedge', 'time', 'player', 'interact', 'real', 'virtual', 'world', 'fellow', 'world', 'player', 'various', 'chat', 'mail', 'fully', 'voiced', 'video', 'news', 'clip', 'together', 'haseos', 'virtual', 'partner', 'login', 'challenge', 'solve', 'mystery', 'new', 'environment', 'world', 'r2', 'second', 'series', 'hack', 'franchise', 'gu', 'feature', 'new', 'familiar', 'character', 'chapter', 'series', 'designed', 'distinct', 'first', 'hack', 'quadrilogy', 'email', 'friend', 'game', 'info', 'developer', 'cyberconnect2', 'publisher', 'bandai', 'genre', 'rpg', 'release', 'date', 'october', '24', '2006', 'esrb', 'teen', 'gamespy', 'score', 'read', 'review', 'cheat', 'walkthroughs', 'strategy', 'guide', 'cheat', 'hackgu', 'vol', '1', 'rebirth', 'article', 'interview', 'hackgu', '092105', 'roundtable', 'discussion', 'tgs', 'bandai', 'reveals', 'whats', 'next', 'hack', 'series', 'movie', 'hackgu', 'vol', '1', 'rebirth', 'trailer', '101906', 'trailer', 'introduce', 'u', 'battle', 'system', 'extraortinary', 'pkk', 'hackgu', 'vol', '1', 'rebirth', 'trailer', '092806', 'new', 'ability', 'shown', 'new', 'trailer', 'namco', 'bandais', 'upcoming', 'dungeon', 'crawler', 'new', 'hackgu', 'part', '1', 'rebirth', 'gameplay', 'video', '081406', 'second', 'series', 'hack', 'franchise', 'hackgu', 'part', '1', 'rebirth', 'new', 'trailer', '060806', 'get', 'new', 'sneak', 'peak', 'second', 'series', 'hack', 'franchise', 'hackgu', '050206', 'awesome', 'new', 'gameplay', 'footage', 'hackgu', 'part', '1', 'trailer', '091505', 'lengthy', 'cinematic', 'teaser', 'preview', 'hackgu', 'vol', '1', 'resurrection', '050206', 'popular', 'offline', 'mmo', 'get', 'new', 'start', 'fresh', 'series', 'hackgu', '091505', 'one', 'inventive', 'rpg', 'series', 'ps2', 'serf', 'tantalizing', 'detail', 'sequel', 'leaf', 'u', 'wanting', 'review', 'hackgu', '102706', 'offline', 'online', 'serial', 'rpg', 'return', 'hold', 'around', 'network', 'hackgu', 'vol', '1', 'ign', 'hackgu', 'vol', '1', 'cheat', 'ign', 'hackgu', 'vol', '1', 'gamespy', 'hackgu', 'vol', '1', 'gamestats', 'hackgu', 'vol', '1', 'cheat', 'ccg', 'hackgu', 'vol', '1', 'askmen']
    # print(term_frequency("1", list))

    # multi_term_tdidf("Hello World Fifa", list)

    ''' --- TO DO ---
        - Get query tf-idf
        - Use Vector Space model
        - Cosine? 
            - both are equal to the dot product slide 50
            - Normalised vectors
        - Stemming ()
        - Clean up webscraper()
        - Read in html files differently
        - CVS look up method for result comparisons
        - Build different engines utilising different components
    '''


main()