import math
import os
import time

import numpy
import pandas
import regex

from bs4 import BeautifulSoup, Comment

import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')
from nltk.tokenize import word_tokenize
# nltk.download('punkt')
from nltk import PorterStemmer
# time.sleep(3) # Ensures all ntlk downloads are up-to-date before system begins

#
# def webScraper():
#     kgrams = []
#
#     # get the name of html file from csv file
#     videogame_details = pandas.read_csv('videogame-labels.csv')
#     videogame_dict = videogame_details.to_dict(orient='records') # Convert CVS file from pandas data structure to dictionary
#
#     # count = 0
#     # for record in videogame_dict:
#     #     print(record)
#     #     count += 1
#     # print(count)
#
#     # get the url from each record reading to scrape, add to list
#     list_of_urls = []
#     for record in videogame_dict:
#         url = record['url']
#         list_of_urls.append(url)
#
#     # print(list_of_urls)
#
#     # Dictionary to store all tokens from one document
#     list_of_document_tokens = []
#
#     # beautiful soup
#     for url in list_of_urls:
#         # remove unneccessary parts of url for html file search
#         pattern = regex.compile(r"^.*?/.*?/") # match from start, any character until first /, then repeat until next one
#         simpleURL = regex.sub(pattern, "", url)
#         # print(simpleURL)
#
#         # r = requests.get('videogames/' + simpleURL)
#         r = open('videogames/' + simpleURL, 'r')
#         soup = BeautifulSoup(r.read(), 'html5lib')
#         soup.prettify()
#         # print(soup)
#
#
#         # -------------------------------------------- REMOVE?
#         # Get game title for kgram/raw name
#         raw_title = soup.find_all("title")
#         game_title = regex.sub(r"GameSpy: ", "", raw_title[0].get_text())
#         kgrams.append(game_title)
#         # print(game_title)
#         # --------------------------------------------
#
#         # Remove css and js from scraped html
#         for data in soup(['style', 'script']):
#             data.decompose()
#
#         # Get all text from under the content id
#         data = soup.find_all(id='content')
#         refined_data = str(data[0].get_text())
#         pattern = regex.compile(r"\s+")
#         simplified_data = regex.sub(pattern, " ", refined_data)
#         # print(simplified_data)
#
#         # --- remove punctuation and lowercase ---
#         removed_punctuation = remove_punc(simplified_data)
#
#         # --- Tokenise ---
#         tokens = tokeniser(removed_punctuation)
#
#         # --- Remove stopwords ---
#         tokened_removed_stopwords = remove_stopwords(tokens)
#
#         # --- Lemmatize ---
#         lemmatized_words = lemmatizer(tokened_removed_stopwords)
#
#         vg_tokens = {"Name": game_title, "Tokens": lemmatized_words}
#         list_of_document_tokens.append(vg_tokens)
#
#     query = "racing action sport"
#
#     # --- Get tf-idf scores for each document based on the query terms ---
#     doc_scores = tfidf(query, list_of_document_tokens)
#
#     # --- Get tf-idf score for the query ---
#     query_scores = query_tfidf(query, list_of_document_tokens)
#
#     # --- Get cosine scores using vector normalisation and dot product ---
#     dp_results = []
#     for q in doc_scores:
#         x = vector_space(query_scores["Vector"], q["Vector"]) # query tfidf scores, doc tfidf scores
#         result_set = {"Name": q["Name"], "Dot product": x}
#         dp_results.append(result_set)
#
#     # --- Cosine comparison (order and precision @10)
#     cosine_similarity(dp_results)
#
#
#     # ---- testing ----
#     # for w in list_of_document_tokens:
#     #     print(w)
#     #     print(w["Tokens"][4])
#     #     if "ps2" in w["Tokens"]:
#     #         print("True")
#
#
#     # testTokens = list_of_document_tokens[0]["Tokens"]
#     # print (testTokens)
#     # x = term_frequency("hackgu", testTokens)
#     # print ("tf ", x)
#     #
#     # print(inverse_document_frequency("wwe", list_of_document_tokens))
#
#     # tfidf_list = tfidf("ICO", list_of_document_tokens)
#     # for i in tfidf_list:
#     #     print(i)
#
#     # query = "soldier ps2 age"
#     query = "racing action sport"
#     multi = tfidf(query, list_of_document_tokens)
#     # # multi = tdidf("soldier", list_of_document_tokens)
#     # for i in multi:
#     #     print(i)
#
#
#     query_scores = query_tfidf(query, list_of_document_tokens)
#     print(query_scores)
#
#
#     # for i in range(len(multi)):
#     #     vector_space(query_scores["Vector"], multi[i]["Vector"]) # query tfidf scores, doc tfidf scores
#
#     dp_results = []
#     for q in multi:
#         # print(q["Name"])
#         # print(q)
#         x = vector_space(query_scores["Vector"], q["Vector"]) # query tfidf scores, doc tfidf scores
#         result_set = {"Name": q["Name"], "Dot product": x}
#         dp_results.append(result_set)
#
#     # for i in dp_results:
#     #     print(i)
#
#     cosine_similarity(dp_results)


def instal_nltk_datasets():
    print("Initalising...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    time.sleep(3) # Ensures all ntlk downloads are up-to-date before system begins

def get_query():
    query = input("Please enter your query: ")
    print("Searching...")
    return query

def query_dealer(query, text_normalisation = 1):
    # Deal with query to get into same format as doc data
    # Remove any punctuation
    revised_query = remove_punc(query)

    # Lemmatise/stemming depending on which the engine uses
    if text_normalisation == 1:
        revised_query = lemmatizer(revised_query.split(" "))
        final_query = ' '.join(map(str, revised_query))
        return final_query

    elif text_normalisation == 2:
        revised_query = stemming(revised_query.split(" "))
        final_query = ' '.join(map(str, revised_query))
        return final_query

    elif text_normalisation == 3:
        return revised_query

def csv_reader():
    # Get the name of html file from csv file
    videogame_details = pandas.read_csv('videogame-labels.csv')
    videogame_dict = videogame_details.to_dict(orient='records')  # Convert CVS file from pandas data structure to dictionary

    for record in videogame_dict:
        print(record)

    # return videogame_dict

def file_accesser():
    file = os.listdir('videogames')
    # print(file)
    file.remove(file[0])  # Remove .DS_store from list of files

    # Create file directory list
    file_list = []
    for i in range(len(file)):
        web_pages = "videogames/" + file[i]
        file_list.append(web_pages)

    return file_list

def web_scraper(paths):
    raw_data = []

    for path in paths:
        r = open(path, 'r')
        soup = BeautifulSoup(r.read(), 'html5lib')
        soup.prettify()
        # print(soup)

        for data in soup(['style', 'script']):
            data.decompose()

        # Get all text from under the content id
        data = soup.find_all(id='content')
        refined_data = str(data[0].get_text())
        pattern = regex.compile(r"\s+")
        simplified_data = regex.sub(pattern, " ", refined_data)

        # print(simplified_data) # Return this
        # raw_data.append(simplified_data)

        # Get game title
        raw_title = soup.find_all("title")
        game_title = regex.sub(r"GameSpy: ", "", raw_title[0].get_text())

        raw_data.append({"Name": game_title, "Data": simplified_data})

    return raw_data

def getGameTitle(path):
    r = open(path, 'r')
    soup = BeautifulSoup(r.read(), 'html5lib')
    soup.prettify()
    raw_title = soup.find_all("title")
    game_title = regex.sub(r"GameSpy: ", "", raw_title[0].get_text())
    return game_title

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
    ps = PorterStemmer()
    stemmed_words = []

    for word in tokens:
        stemmed_words.append(ps.stem(word))

    return stemmed_words

def term_frequency(term, doc_tokens):
    # if no term in doc, score = 0
    # tf = 1 + math.log(tf)

    frequency = 0
    for i in doc_tokens:
        if term == i:
            frequency += 1

    # .count for frequency?

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
        # else:
        #     continue # Continue until term occurs or not found

    collection_size = len(vg_list)

    # Handle if term does not appear in any document
    if document_frequency == 0:
        return 0
    else:
        idf = math.log((collection_size / document_frequency), 10)
        return idf

def tfidf(query, vg_list):
    query_terms = query.split(" ") # ------------ change to use tokeniser
    # query_terms = tokeniser(query) # ------------ change to use tokeniser
    # print(query_terms)


    docs_tfidf = []

    for i in range(len(vg_list)):

        term_tfidf_list = []
        tfidf_list = []

        for j in range(len(query_terms)):
            tfidf = term_frequency(query_terms[j], vg_list[i]["Tokens"]) * inverse_document_frequency(query_terms[j], vg_list)
            # print(tfidf)

            t = {"Term": query_terms[j], "tf-idf": tfidf}
            term_tfidf_list.append(t)
            tfidf_list.append(tfidf)

        all_terms_tfidf = {"Name": vg_list[i]["Name"], "Vector": tfidf_list, "Query-term tf-idf": term_tfidf_list}
        docs_tfidf.append(all_terms_tfidf)

    return docs_tfidf

def query_tfidf(query, vg_list):
    # Create tf-idf for the query
    query_tokens = tokeniser(remove_punc(query)) # Remove punctuation and make lower case to align with text data
    term_tfidf = []
    vector = []

    for term in query_tokens:
        tf = term_frequency(term, query_tokens) # count number of times each term appears in query, divided by total number of terms in query
        idf = inverse_document_frequency(term, vg_list)
        tfidf = tf * idf
        t = {"Term": term, "tf-idf": tfidf}
        term_tfidf.append(t)
        vector.append(tfidf)

    query_tfidf = {"Terms": term_tfidf, "Vector": vector}

    return query_tfidf

def vector_space(query_tfidf, vg_docs_tfidf):
    # print("Vectors")
    # Length normalised vectors allow for cosine comparison (dot product)
    normalised_query_v = vector_len_normalisation(query_tfidf)
    normalised_docs_v = vector_len_normalisation(vg_docs_tfidf)
    # print("normalised query ", normalised_query_v)
    # print("normalised doc ", normalised_docs_v)

    # Dot product
    dot_product = numpy.dot(normalised_query_v, normalised_docs_v)
    # print("dp", dot_product)
    return float(dot_product)

def vector_len_normalisation(vector):
    # Divide each term/component by the vector magnitude
    # Normalised vector lengths are always 1
    magnitude = math.sqrt(sum([i**2 for i in vector]))
    # print("vector len normalisation ", magnitude)
    if magnitude == 0:
        # return an array of len(vector) of 0s for ease of dot product calculation
        return [0 for l in vector]

    else:
        len_normalised = [x / magnitude for x in vector]
    # print(len_normalised)
    return len_normalised

def cosine_similarity(dp_result_set):
    # Sort out result set of the dot product calculation into desc order
    sorted_rs = sorted(dp_result_set, key=lambda x:x['Dot product'], reverse = True)

    # @10 precision
    for i in range(0, 10):
        print(sorted_rs[i])

    # Press key to continue after results displayed
    input("Press enter to continue...")

def test_engine(query):
    # --- Testing

    # --- Retrieve filenames ---
    files = file_accesser()

    # --- Scrape the website for content ---
    raw_data = web_scraper(files)
    document_tokens = []

    for data in raw_data:
        # --- remove punctuation and lowercase ---
        removed_punctuation = remove_punc(data["Data"])

        # --- Tokenise ---
        tokens = tokeniser(removed_punctuation)

        # --- Remove stopwords ---
        tokened_removed_stopwords = remove_stopwords(tokens)

        # --- Lemmatize ---
        lemmatized_words = lemmatizer(tokened_removed_stopwords)

        # --- Stemming ---
        stemming(tokened_removed_stopwords)

        vg_tokens = {"Name": data["Name"], "Tokens": lemmatized_words}
        document_tokens.append(vg_tokens)

    # print(document_tokens)
    # --- Get tf-idf scores for each doc based on the query terms ---
    doc_scores = tfidf(query, document_tokens)

    # --- Get tf-idf score for the query ---
    query_scores = query_tfidf(query, document_tokens)

    # --- Get cosine scores using vector normalisation and dot product ---
    dp_results = []
    for q in doc_scores:
        dp = vector_space(query_scores["Vector"], q["Vector"])
        result_set = {"Name": q["Name"], "Dot product": dp}
        dp_results.append(result_set)

    # --- Cosine comparison (order & precision @ 10)
    cosine_similarity(dp_results)


def main():
    # query = getQuery()
    # print("Your query is:", query)
    # webScraper()
    # csv_reader()

    instal_nltk_datasets()
    # query = query_dealer(get_query())
    # test_engine(query)

    lm = WordNetLemmatizer()
    print(lm.lemmatize("smallest"))
    ps = PorterStemmer()
    print(ps.stem("smallest"))




    ''' --- TO DO ---
        - CVS look up method for result comparisons
        - Build different engines utilising different components
        - need to lem/stem the query to get to the same simplification as the docment text
        - named entity recognition
        - Query()
        - README file
        - only return games with a score higher than 0
        - "press enter to continue" message after search is complete/results are displayed
        - jarcard?
        
    '''


# main()