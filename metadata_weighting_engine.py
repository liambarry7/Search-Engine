import math
import os
import time

import numpy
import pandas
import pandas as pd
import spacy
import regex

from bs4 import BeautifulSoup, Comment
import engine_components as ec


def csv_reader(game):
    # Get the name of html file from csv file
    videogame_details = pandas.read_csv('videogame-labels.csv')
    # Convert CVS file from pandas data structure to dictionary
    videogame_dict = videogame_details.to_dict(orient='records')

    for record in videogame_dict:
        if record['url'] == 'videogame/ps2.gamespy.com/' + game:
            # print(record)
            return record

    # return videogame_dict

def get_metadata(game):
    try:
        game['url'] = game["url"].split("/")[-1].replace('-', ' ').replace('.html', '').lower()
        values = list(game.values())
        values = ' '.join(map(str, values))  # all but first value
        values = ec.remove_punc(values)
        values = values.split(" ")
        return values
        # print(values)

    except Exception as e:
        # capture errors such as path not existing in csv
        # print(f"Error: {e}")
        return ''

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


        # get metadata from csv
        #remove videogames/ from paths
        # print(path)
        simple_path = path.replace('videogames/', '')
        metadata = get_metadata(csv_reader(simple_path))

        raw_data.append({"Name": game_title, "Metadata": metadata, "Data": simplified_data})


    return raw_data


def term_frequency(term, doc_tokens):
    # if no term in doc, score = 0
    # tf = 1 + math.log(tf)

    frequency = 0
    for i in doc_tokens:
        if term == i:
            frequency += 1

    if (frequency > 0):
        tf = 1 + math.log(frequency, 10)  # log x, base 10
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
                break  # Break inner loop as term already found once in doc
        # else:
        #     continue # Continue until term occurs or not found

    collection_size = len(vg_list)

    # Handle if term does not appear in any document
    if document_frequency == 0:
        return 0
    else:
        idf = math.log((collection_size / document_frequency), 10)
        return idf


def tfidf(expanded_query, vg_list):
    query_terms = expanded_query.split(" ")  # ------------ change to use tokeniser
    # query_terms = tokeniser(query) # ------------ change to use tokeniser

    docs_tfidf = []

    for i in range(len(vg_list)):

        term_tfidf_list = []
        tfidf_list = []

        for j in range(len(query_terms)):
            tfidf = term_frequency(query_terms[j], vg_list[i]["Tokens"]) * inverse_document_frequency(query_terms[j], vg_list)

            weighted_tfidf = weight_booster(tfidf, vg_list[i]["Metadata"], query_terms[j])  # i is the index


            t = {"Term": query_terms[j], "tf-idf": weighted_tfidf}
            # t = {"Term": query_terms[j], "tf-idf": tfidf}
            term_tfidf_list.append(t)
            # tfidf_list.append(tfidf)
            tfidf_list.append(weighted_tfidf)

        all_terms_tfidf = {"Name": vg_list[i]["Name"], "Vector": tfidf_list, "Query-term tf-idf": term_tfidf_list}
        docs_tfidf.append(all_terms_tfidf)

    return docs_tfidf


def query_tfidf(query, vg_list):
    # Create tf-idf for the query
    query_tokens = ec.tokeniser(ec.remove_punc(query))  # Remove punctuation and make lower case to align with text data
    term_tfidf = []
    vector = []

    for term in query_tokens:
        tf = term_frequency(term,
                            query_tokens)  # count number of times each term appears in query, divided by total number of terms in query
        idf = inverse_document_frequency(term, vg_list)
        tfidf = tf * idf

        weighted_tfidf = weight_booster(tfidf, query_tokens, term)


        # t = {"Term": term, "tf-idf": tfidf}
        t = {"Term": term, "tf-idf": weighted_tfidf}
        term_tfidf.append(t)
        # vector.append(tfidf)
        vector.append(weighted_tfidf)

    query_tfidf = {"Terms": term_tfidf, "Vector": vector}

    return query_tfidf


def weight_booster(tfidf, metadata, query):
    new_weight = tfidf # Return original weight if query not in metadata
    for term in metadata:
        if query == term:
            new_weight = tfidf * 10

    return new_weight
