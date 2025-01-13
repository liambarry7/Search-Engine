import math
import os
import time

import numpy
import pandas
import pandas as pd
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
    time.sleep(3)  # Ensures all ntlk downloads are up-to-date before system begins


def get_query():
    query = input("Please enter your query: ")
    print("Searching...")
    return query


def query_dealer(query, text_normalisation=1):
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


def synonym_expansion(query):
    expanded_query = set()  # Use set as no duplicates allowed

    for term in query.split():
        expanded_query.add(term)  # Add the original term back to query
        for synonym in wordnet.synsets(term):
            for word in synonym.lemmas():
                expanded_query.add(word.name().replace('_', ' '))

    return ' '.join(map(str, expanded_query))


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

def get_metadataX(game):
    # print(game)
    try:
        game['url'] = game["url"].split("/")[-1].replace('-', ' ').replace('.html', '').lower()
        values = list(game.values())
        values = ' '.join(map(str, values))  # all but first value
        values = remove_punc(values)
        values = values.split(" ")
        return values
        # print(values)

    except Exception as e:
        # capture errors such as path not existing in csv
        print(f"Error: {e}")
        return ''


def get_metadata():
    videogame_details = pandas.read_csv('videogame-labels.csv')
    # Convert CVS file from pandas data structure to dictionary
    videogame_dict = videogame_details.to_dict(
        orient='records')

    documents_metadata = []
    for record in videogame_dict:
        # name = record["url"].split("/")[-1].replace('-', ' ').replace('.html', '').lower()
        record['url'] = record["url"].split("/")[-1].replace('-', ' ').replace('.html', '').lower()
        # record.pop("url")
        # record["Name"] = name
        documents_metadata.append(record)

        # documents_metadata.append((', '.join(map(str, record.values()))).replace(',', ''))
        # pure_data = (', '.join(map(str, record.values()))).replace(',', '').lower()
        # dict_data = {"Name": name, "Metadata": pure_data}
        # documents_metadata.append(dict_data)

    # Sort list
    sorted_docs = sorted(documents_metadata, key=lambda doc: (not doc['url'][0].isdigit(), doc['url']))
    # return documents_metadata
    # print(sorted_docs)
    return sorted_docs


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
        metadata = get_metadataX(csv_reader(simple_path))

        raw_data.append({"Name": game_title, "Metadata": metadata, "Data": simplified_data})


    return raw_data


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

    # documents_metadata = get_metadata()
    # for i in documents_metadata:

    # frequency = 0
    # for i in doc_tokens:
    #     for metadata in list(documents_metadata):
    # print(metadata)

    # if term == i: # in query
    #     if name == metadata["Name"]:
    #         for j in metadata["Metadata"].split(" "):
    #             if j == term: # in metadata
    #                 frequency += 1
    # else:
    #     frequency += 1

    # .count for frequency?

    if (frequency > 0):
        tf = 1 + math.log(frequency, 10)  # log x, base 10
        return tf

    else:
        return 0


# def query_term_frequency(term, doc_tokens):
# if no term in doc, score = 0
# tf = 1 + math.log(tf)

# frequency = 0
# for i in doc_tokens:
#     if term == i:
#         frequency += 1
#
# if (frequency > 0):
#     tf = 1 + math.log(frequency, 10) # log x, base 10
#     return tf
#
# else:
#     return 0

# documents_metadata = get_metadata()
# # for i in documents_metadata:
#
# frequency = 0
# for i in doc_tokens:
#     for metadata in list(documents_metadata):
#         # print(metadata)
#
#         if term == i:  # in query
#
#         #         for j in metadata["Metadata"].split(" "):
#         #             if j == term:  # in metadata
#         #                 frequency += 1
#         # else:
#             frequency += 1
#
# if (frequency > 0):
#     tf = 1 + math.log(frequency, 10) # log x, base 10
#     return tf
#
# else:
#     return 0

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
    # print(query_terms)

    docs_tfidf = []

    for i in range(len(vg_list)):

        term_tfidf_list = []
        tfidf_list = []

        for j in range(len(query_terms)):
            tfidf = term_frequency(query_terms[j], vg_list[i]["Tokens"]) * inverse_document_frequency(query_terms[j], vg_list)
            # print("vg", vg_list[i]["Name"])
            # print("tfidf", tfidf)


            # for data in get_metadata():
            # weight_booster(tfidf, document_tokens, query) # not with synonyms to keep context of query
            # weight_booster(tfidf, vg_list[i], query) # not with synonyms to keep context of query
            weighted_tfidf = weight_booster(tfidf, vg_list[i]["Metadata"], query_terms[j])  # i is the index
            # print(weighted_tfidf)

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
    query_tokens = tokeniser(remove_punc(query))  # Remove punctuation and make lower case to align with text data
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
    # print("Boost")
    # print(f"query {query}")
    new_weight = 0
    for term in metadata:
        # print(f"term {term}")
        if query == term:
            new_weight = tfidf * 1.5
            # print("Succ")

    return new_weight

    '''
    # metadata = get_metadata()
    # print(metadata)

    # query_terms = remove_stopwords(query.split(" "))
    # already individual query term

    mulitplier = 0
    # for metadata in get_metadata():
    #     print(metadata)
    #     # result = metadata.values()
    #     result = ' '.join(map(str, metadata.values()))
    #     result = remove_punc(result)
    #     result = result.split(" ")
    #     # result = ' '.join(map(str, remove_punc(metadata.values()))) # turn into list of strings
    #     print(result)
    # print(len(metadata))
    # print(index)
    data = metadata[index - 1]
    print("meta", data["url"], "\n")

    results = ' '.join(map(str, data.values()))
    results = remove_punc(results)
    results = results.split(" ")
    # print(results)
    # print(query)

    weighted_tfidf = 0
    for term in results:
        if term == query:
            weighted_tfidf = tfidf * 10

    # print(weighted_tfidf)
    return weighted_tfidf

    # for data in metadata[index-1]:
    #     print(data)

    # result = data.values()
    # result = ' '.join(map(str, metadata.values()))
    # result = remove_punc(result)
    # result = result.split(" ")
    # # result = ' '.join(map(str, remove_punc(metadata.values()))) # turn into list of strings
    # print(result)

    # for key, value in metadata.items():
    #     print(value)
    # if query in value:
    #     print("Query in metadata")
    # data = str(value).lower().split(" ")
    # print(value)
    # print(remove_punc(str(value)))
    # metadata_values = remove_punc(str(value)).split(" ")
    # for x in remove_punc(str(value)).split(" "):
    # common_terms = [term for term in query_terms if term in metadata_values]
    # print("common terms", common_terms)
    # mulitplier = mulitplier + len(common_terms) # add no of common terms to multipler

    # for data in metadata:
    #     result = ' '.join(map(str, data.values()))
    #     result = remove_punc(result).lower()
    #     meta = result.split(" ")
    #     q = query.split(" ")
    #     q = remove_stopwords(q)
    #     # print(result)
    #
    #     for term in q:
    #         for i in meta:
    #             if term == i:
    #                 print(result)

    # for vg in vg_list:
    #     name = remove_punc(vg["Name"]).lower()
    #     print(name)
    #     # for data in metadata:
    #     #     # print(data["Name"])
    #     #     if name == data["url"]:
    #     #         print(data)
    '''


def vector_space(query_tfidf, vg_docs_tfidf):
    # print("Vectors")
    # Length normalised vectors allow for cosine comparison (dot product)
    normalised_query_v = vector_len_normalisation(query_tfidf)
    normalised_docs_v = vector_len_normalisation(vg_docs_tfidf)

    # Dot product
    dot_product = numpy.dot(normalised_query_v, normalised_docs_v)
    return float(dot_product)


def vector_len_normalisation(vector):
    # Divide each term/component by the vector magnitude
    # Normalised vector lengths are always 1
    magnitude = math.sqrt(sum([i ** 2 for i in vector]))
    if magnitude == 0:
        # return an array of len(vector) of 0s for ease of dot product calculation
        return [0 for l in vector]

    else:
        len_normalised = [x / magnitude for x in vector]
    return len_normalised


def cosine_similarity(dp_result_set, game_desc):
    # Sort out result set of the dot product calculation into desc order
    sorted_rs = sorted(dp_result_set, key=lambda x: x['Dot product'], reverse=True)

    # @10 precision
    for i in range(0, 10):
        # Match the game titles to find the correct description
        for j in range(len(game_desc)):
            if sorted_rs[i]["Name"] == game_desc[j]["Name"]:
                # print("{} - {}:{}... ".format(i+1, sorted_rs[i]["Name"], game_desc[j]["Data"][:100]))
                print("{} - {}:{}... dp:{}".format(i + 1, sorted_rs[i]["Name"], game_desc[j]["Data"][:100],
                                                   sorted_rs[i]["Dot product"]))

    precision = sum(1 for x in sorted_rs[:10] if x["Dot product"] > 0)
    print(f"Precision @10: {precision}")

    # Press key to continue after results displayed
    x = input("Press enter to continue...")


def named_entity_relation(document):
    # TEST METHOD
    # https://www.geeksforgeeks.org/named-entity-recognition/
    # use spacy
    # Add to document tokens
    nlp = spacy.load("en_core_web_sm")  # Load english spacy model

    document_entities = set()  # Empty set to hold each name once

    content = nlp(document['Data'])  # Process text
    for ent in content.ents:  # Extract entities
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

        vg_tokens = {"Name": data["Name"], "Metadata": data["Metadata"], "Tokens": lemmatized_words}
        # vg_tokens = {"Name": data["Name"], "Tokens": lemmatized_words, "Entities": named_entity_relation(data)}
        document_tokens.append(vg_tokens)

    # print(document_tokens)

    # weight_booster(1.4, document_tokens, query) # not with synonyms to keep context of query

    # --- Get tf-idf scores for each doc based on the query terms ---
    doc_scores = tfidf(expanded_query, document_tokens)
    # doc_scores = tfidf(expanded_query, document_tokens, query)
    for i in doc_scores:
        print(i)

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
    # query = getQuery()
    # print("Your query is:", query)
    # webScraper()
    # csv_reader()

    # instal_nltk_datasets()
    # query = query_dealer(get_query())
    test_engine()

    # weight_booster(4.32)
    # a = file_accesser()
    # b = web_scraper(a)
    # for x in b:
    # print(named_entity_relation(x))
    # print(x)

    # named_entity_relation(b)

    # lm = WordNetLemmatizer()
    # print(lm.lemmatize("smallest"))
    # ps = PorterStemmer()
    # print(ps.stem("smallest"))

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
    # synonym_expansion("run joggers")


main()