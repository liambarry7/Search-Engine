import math

import spacy
import engine_components as ec

def named_entity_relation(document):
    # print("NER")
    # https://www.geeksforgeeks.org/named-entity-recognition/
    # use spacy
    # Add to document tokens
    nlp = spacy.load("en_core_web_sm") # Load english spacy model

    document_entities = set() # Empty set to hold each name once

    content = nlp(document['Data']) # Process text
    for ent in content.ents: # Extract entities
        document_entities.add((ent.text).lower())

    return ', '.join(map(str, document_entities))

def term_frequency(term, doc_tokens, doc_entities):
    # if no term in doc, score = 0
    # tf = 1 + math.log(tf)

    frequency = 0
    for i in doc_tokens:
        if term == i:
            frequency += 1

    for j in doc_entities:
        if term == j:
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


    # Basic df check
    document_frequency = 0
    for vg in vg_list:
        for token in vg["Tokens"]:
            if token == term:
                document_frequency += 1
                break # Break inner loop as term already found once in doc

        for token in vg["Entities"]:
            if token == term:
                document_frequency += 1
                break

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


    docs_tfidf = []

    for i in range(len(vg_list)):

        term_tfidf_list = []
        tfidf_list = []

        for j in range(len(query_terms)):
            tfidf = term_frequency(query_terms[j], vg_list[i]["Tokens"], vg_list[i]["Entities"]) * inverse_document_frequency(query_terms[j], vg_list)

            t = {"Term": query_terms[j], "tf-idf": tfidf}
            term_tfidf_list.append(t)
            tfidf_list.append(tfidf)

        all_terms_tfidf = {"Name": vg_list[i]["Name"], "Vector": tfidf_list, "Query-term tf-idf": term_tfidf_list}
        docs_tfidf.append(all_terms_tfidf)

    return docs_tfidf

def query_tfidf(query, vg_list):
    # Create tf-idf for the query
    query_tokens = ec.tokeniser(ec.remove_punc(query)) # Remove punctuation and make lower case to align with text data
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