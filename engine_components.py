import math
import os
import time

import numpy
import spacy
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
from nltk.corpus import wordnet

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
        final_query = ' '.join(map(str, revised_query)) # https://www.simplilearn.com/tutorials/python-tutorial/list-to-string-in-python#:~:text=To%20convert%20a%20list%20to%20a%20string%2C%20use%20Python%20List,and%20return%20it%20as%20output
        return final_query

    elif text_normalisation == 2:
        revised_query = stemming(revised_query.split(" "))
        final_query = ' '.join(map(str, revised_query))
        return final_query

    elif text_normalisation == 3:
        return revised_query

def synonym_expansion(query):
    # https://www.geeksforgeeks.org/get-synonymsantonyms-nltk-wordnet-python/

    expanded_query = set() # Use set as no duplicates allowed

    for term in query.split():
        expanded_query.add(term) # Add the original term back to query
        for synonym in wordnet.synsets(term):
            for word in synonym.lemmas():
                expanded_query.add(word.name().replace('_', ' '))

    return ' '.join(map(str, expanded_query))


def file_accesser():
    file = os.listdir('videogames')
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

        # Remove all css and javacript from data
        for data in soup(['style', 'script']):
            data.decompose()

        # Get all text from under the content id
        data = soup.find_all(id='content')
        refined_data = str(data[0].get_text())
        pattern = regex.compile(r"\s+")
        simplified_data = regex.sub(pattern, " ", refined_data)

        # Get game title
        raw_title = soup.find_all("title")
        game_title = regex.sub(r"GameSpy: ", "", raw_title[0].get_text())

        raw_data.append({"Name": game_title, "Data": simplified_data})

    return raw_data

def remove_punc(text_data):
    # Removing punctuation from text
    removed_punctation = regex.sub(r"[^\w\s]", "", text_data)
    removed_punctation = removed_punctation.lower()
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

    # Basic df check
    document_frequency = 0
    for vg in vg_list:
        for token in vg["Tokens"]:
            if token == term:
                document_frequency += 1
                break # Break inner loop as term already found once in doc

    collection_size = len(vg_list)

    # Handle if term does not appear in any document
    if document_frequency == 0:
        return 0
    else:
        idf = math.log((collection_size / document_frequency), 10)
        return idf

def tfidf(expanded_query, vg_list):
    # https://courses.cs.washington.edu/courses/cse373/17au/project3/project3-2.html
    query_terms = expanded_query.split(" ") # ------------ change to use tokeniser
    # query_terms = tokeniser(query) # ------------ change to use tokeniser

    docs_tfidf = []

    for i in range(len(vg_list)):

        term_tfidf_list = []
        tfidf_list = []

        for j in range(len(query_terms)):
            tfidf = term_frequency(query_terms[j], vg_list[i]["Tokens"]) * inverse_document_frequency(query_terms[j], vg_list)

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
    # https://numpy.org/doc/2.1/reference/generated/numpy.dot.html
    # Length normalised vectors allow for cosine comparison (dot product)
    normalised_query_v = vector_len_normalisation(query_tfidf)
    normalised_docs_v = vector_len_normalisation(vg_docs_tfidf)

    # Dot product
    dot_product = numpy.dot(normalised_query_v, normalised_docs_v)
    return float(dot_product)

def vector_len_normalisation(vector):
    # Divide each term/component by the vector magnitude
    # Normalised vector lengths are always 1
    magnitude = math.sqrt(sum([i**2 for i in vector]))
    if magnitude == 0:
        # return an array of len(vector) of 0s for ease of dot product calculation
        return [0 for l in vector]

    else:
        len_normalised = [x / magnitude for x in vector]
    return len_normalised

def cosine_similarity(dp_result_set, game_desc):
    # Sort out result set of the dot product calculation into desc order
    sorted_rs = sorted(dp_result_set, key=lambda x:x['Dot product'], reverse = True)

    # @10 precision
    for i in range(0, 10):
        # Match the game titles to find the correct description
        for j in range(len(game_desc)):
            if sorted_rs[i]["Name"] == game_desc[j]["Name"]:
                # print("{} - {}:{}... ".format(i+1, sorted_rs[i]["Name"], game_desc[j]["Data"][:100]))
                print("{} - {}:{}... dp:{}".format(i+1, sorted_rs[i]["Name"], game_desc[j]["Data"][:50], sorted_rs[i]["Dot product"]))

    precision = sum(1 for x in sorted_rs[:10] if x["Dot product"] > 0)
    print(f"Precision @10: {precision}")

    # Press key to continue after results displayed
    x = input("Press enter to continue...")

def named_entity_relation(document):
    # TEST METHOD
    # https://www.geeksforgeeks.org/named-entity-recognition/
    # use spacy
    # Add to document tokens
    nlp = spacy.load("en_core_web_sm") # Load english spacy model

    document_entities = set() # Empty set to hold each name once

    content = nlp(document['Data']) # Process text
    for ent in content.ents: # Extract entities
        document_entities.add((ent.text).lower())

    return ', '.join(map(str, document_entities))


def test_engine():
    # --- Testing
    query = query_dealer(get_query(), 1)
    expanded_query = synonym_expansion(query)

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
        # vg_tokens = {"Name": data["Name"], "Tokens": lemmatized_words, "Entities": named_entity_relation(data)}
        document_tokens.append(vg_tokens)

    # print(document_tokens)

    # weight_booster(1.4, document_tokens, query) # not with synonyms to keep context of query


    # --- Get tf-idf scores for each doc based on the query terms ---
    doc_scores = tfidf(expanded_query, document_tokens)
    # doc_scores = tfidf(expanded_query, document_tokens, query)
    # print(doc_scores)


    # --- Get tf-idf score for the query ---
    query_scores = query_tfidf(expanded_query, document_tokens)

    # --- Get cosine scores using vector normalisation and dot product ---
    dp_results = []
    for q in doc_scores:
        dp = vector_space(query_scores["Vector"], q["Vector"])
        result_set = {"Name": q["Name"], "Dot product": dp}
        dp_results.append(result_set)

    # --- Cosine comparison (order & precision @ 10)
    cosine_similarity(dp_results, raw_data)


def main():
    test_engine()
    ''' --- TO DO ---
        - named entity recognition
        - metadata?
        - bigrams of metadata and query?
        - README file
        - use matlib to generate graphs of dp
        
        Links:
        https://www.geeksforgeeks.org/get-synonymsantonyms-nltk-wordnet-python/
        https://courses.cs.washington.edu/courses/cse373/17au/project3/project3-2.html
        https://numpy.org/doc/2.1/reference/generated/numpy.dot.html
        https://www.simplilearn.com/tutorials/python-tutorial/list-to-string-in-python#:~:text=To%20convert%20a%20list%20to%20a%20string%2C%20use%20Python%20List,and%20return%20it%20as%20output.
        
        
    '''

# main()