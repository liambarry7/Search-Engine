import engine_components as ec


def enginex():
    print("Engine 1: Full components")
    return ec.query_dealer(ec.get_query())


def engine1():
    # --- Lemmatisation
    query = ec.query_dealer(ec.get_query(), 1)

    # --- Retrieve filenames ---
    files = ec.file_accesser()

    # --- Scrape the website for content ---
    raw_data = ec.web_scraper(files)
    document_tokens = []

    for data in raw_data:
        # --- remove punctuation and lowercase ---
        removed_punctuation = ec.remove_punc(data["Data"])

        # --- Tokenise ---
        tokens = ec.tokeniser(removed_punctuation)

        # --- Remove stopwords ---
        tokened_removed_stopwords = ec.remove_stopwords(tokens)

        # --- Lemmatize ---
        lemmatized_words = ec.lemmatizer(tokened_removed_stopwords)

        # --- Stemming ---
        # ec.stemming(tokened_removed_stopwords)

        vg_tokens = {"Name": data["Name"], "Tokens": lemmatized_words}
        document_tokens.append(vg_tokens)

    # print(document_tokens)
    # --- Get tf-idf scores for each doc based on the query terms ---
    doc_scores = ec.tfidf(query, document_tokens)

    # --- Get tf-idf score for the query ---
    query_scores = ec.query_tfidf(query, document_tokens)

    # --- Get cosine scores using vector normalisation and dot product ---
    dp_results = []
    for q in doc_scores:
        dp = ec.vector_space(query_scores["Vector"], q["Vector"])
        result_set = {"Name": q["Name"], "Dot product": dp}
        dp_results.append(result_set)

    # --- Cosine comparison (order & precision @ 10)
    ec.cosine_similarity(dp_results)

def engine2():
    # --- Stemming
    query = ec.query_dealer(ec.get_query(), 2)

    # --- Retrieve filenames ---
    files = ec.file_accesser()

    # --- Scrape the website for content ---
    raw_data = ec.web_scraper(files)
    document_tokens = []

    for data in raw_data:
        # --- remove punctuation and lowercase ---
        removed_punctuation = ec.remove_punc(data["Data"])

        # --- Tokenise ---
        tokens = ec.tokeniser(removed_punctuation)

        # --- Remove stopwords ---
        tokened_removed_stopwords = ec.remove_stopwords(tokens)

        # --- Lemmatize ---
        # lemmatized_words = ec.lemmatizer(tokened_removed_stopwords)

        # --- Stemming ---
        stemmed_words = ec.stemming(tokened_removed_stopwords)

        vg_tokens = {"Name": data["Name"], "Tokens": stemmed_words}
        document_tokens.append(vg_tokens)

    # print(document_tokens)
    # --- Get tf-idf scores for each doc based on the query terms ---
    doc_scores = ec.tfidf(query, document_tokens)

    # --- Get tf-idf score for the query ---
    query_scores = ec.query_tfidf(query, document_tokens)

    # --- Get cosine scores using vector normalisation and dot product ---
    dp_results = []
    for q in doc_scores:
        dp = ec.vector_space(query_scores["Vector"], q["Vector"])
        result_set = {"Name": q["Name"], "Dot product": dp}
        dp_results.append(result_set)

    # --- Cosine comparison (order & precision @ 10)
    ec.cosine_similarity(dp_results)

def engine3():
    # No stemming or lemmatisation
    query = ec.query_dealer(ec.get_query(), 3)

    # --- Retrieve filenames ---
    files = ec.file_accesser()

    # --- Scrape the website for content ---
    raw_data = ec.web_scraper(files)
    document_tokens = []

    for data in raw_data:
        # --- remove punctuation and lowercase ---
        removed_punctuation = ec.remove_punc(data["Data"])

        # --- Tokenise ---
        tokens = ec.tokeniser(removed_punctuation)

        # --- Remove stopwords ---
        tokened_removed_stopwords = ec.remove_stopwords(tokens)

        # # --- Lemmatize ---
        # lemmatized_words = ec.lemmatizer(tokened_removed_stopwords)

        # --- Stemming ---
        # ec.stemming(tokened_removed_stopwords)

        vg_tokens = {"Name": data["Name"], "Tokens": tokened_removed_stopwords}
        document_tokens.append(vg_tokens)

    # print(document_tokens)
    # --- Get tf-idf scores for each doc based on the query terms ---
    doc_scores = ec.tfidf(query, document_tokens)

    # --- Get tf-idf score for the query ---
    query_scores = ec.query_tfidf(query, document_tokens)

    # --- Get cosine scores using vector normalisation and dot product ---
    dp_results = []
    for q in doc_scores:
        dp = ec.vector_space(query_scores["Vector"], q["Vector"])
        result_set = {"Name": q["Name"], "Dot product": dp}
        dp_results.append(result_set)

    # --- Cosine comparison (order & precision @ 10)
    ec.cosine_similarity(dp_results)

def engine4():
    # Stopwords remaining
    query = ec.query_dealer(ec.get_query(), 1)

    # --- Retrieve filenames ---
    files = ec.file_accesser()

    # --- Scrape the website for content ---
    raw_data = ec.web_scraper(files)
    document_tokens = []

    for data in raw_data:
        # --- remove punctuation and lowercase ---
        removed_punctuation = ec.remove_punc(data["Data"])

        # --- Tokenise ---
        tokens = ec.tokeniser(removed_punctuation)

        # --- Remove stopwords ---
        # tokened_removed_stopwords = ec.remove_stopwords(tokens)

        # --- Lemmatize ---
        lemmatized_words = ec.lemmatizer(tokens)

        vg_tokens = {"Name": data["Name"], "Tokens": lemmatized_words}
        document_tokens.append(vg_tokens)

    # print(document_tokens)
    # --- Get tf-idf scores for each doc based on the query terms ---
    doc_scores = ec.tfidf(query, document_tokens)

    # --- Get tf-idf score for the query ---
    query_scores = ec.query_tfidf(query, document_tokens)

    # --- Get cosine scores using vector normalisation and dot product ---
    dp_results = []
    for q in doc_scores:
        dp = ec.vector_space(query_scores["Vector"], q["Vector"])
        result_set = {"Name": q["Name"], "Dot product": dp}
        dp_results.append(result_set)

    # --- Cosine comparison (order & precision @ 10)
    ec.cosine_similarity(dp_results)

def engine5():
    # Punctuation remaining
    query = ec.query_dealer(ec.get_query(), 1)

    # --- Retrieve filenames ---
    files = ec.file_accesser()

    # --- Scrape the website for content ---
    raw_data = ec.web_scraper(files)
    document_tokens = []

    for data in raw_data:
        # --- remove punctuation and lowercase ---
        # removed_punctuation = ec.remove_punc(data["Data"])
        lowercase_data = data["Data"].lower()
        # --- Tokenise ---
        tokens = ec.tokeniser(lowercase_data)

        # --- Remove stopwords ---
        tokened_removed_stopwords = ec.remove_stopwords(tokens)

        # --- Lemmatize ---
        lemmatized_words = ec.lemmatizer(tokened_removed_stopwords)

        vg_tokens = {"Name": data["Name"], "Tokens": lemmatized_words}
        document_tokens.append(vg_tokens)

    # print(document_tokens)
    # --- Get tf-idf scores for each doc based on the query terms ---
    doc_scores = ec.tfidf(query, document_tokens)

    # --- Get tf-idf score for the query ---
    query_scores = ec.query_tfidf(query, document_tokens)

    # --- Get cosine scores using vector normalisation and dot product ---
    dp_results = []
    for q in doc_scores:
        dp = ec.vector_space(query_scores["Vector"], q["Vector"])
        result_set = {"Name": q["Name"], "Dot product": dp}
        dp_results.append(result_set)

    # --- Cosine comparison (order & precision @ 10)
    ec.cosine_similarity(dp_results)

def engine6():
    # No Query Expansion
    query = ec.query_dealer(ec.get_query(), 1)

    # --- Retrieve filenames ---
    files = ec.file_accesser()

    # --- Scrape the website for content ---
    raw_data = ec.web_scraper(files)
    document_tokens = []

    for data in raw_data:
        # --- remove punctuation and lowercase ---
        removed_punctuation = ec.remove_punc(data["Data"])

        # --- Tokenise ---
        tokens = ec.tokeniser(removed_punctuation)

        # --- Remove stopwords ---
        tokened_removed_stopwords = ec.remove_stopwords(tokens)

        # --- Lemmatize ---
        lemmatized_words = ec.lemmatizer(tokened_removed_stopwords)

        vg_tokens = {"Name": data["Name"], "Tokens": lemmatized_words}
        document_tokens.append(vg_tokens)

    # print(document_tokens)
    # --- Get tf-idf scores for each doc based on the query terms ---
    doc_scores = ec.tfidf(query, document_tokens)

    # --- Get tf-idf score for the query ---
    query_scores = ec.query_tfidf(query, document_tokens)

    # --- Get cosine scores using vector normalisation and dot product ---
    dp_results = []
    for q in doc_scores:
        dp = ec.vector_space(query_scores["Vector"], q["Vector"])
        result_set = {"Name": q["Name"], "Dot product": dp}
        dp_results.append(result_set)

    # --- Cosine comparison (order & precision @ 10)
    ec.cosine_similarity(dp_results)

def engine7():
    # No Named Entity Recognition
    query = ec.query_dealer(ec.get_query(), 1)

    # --- Retrieve filenames ---
    files = ec.file_accesser()

    # --- Scrape the website for content ---
    raw_data = ec.web_scraper(files)
    document_tokens = []

    for data in raw_data:
        # --- remove punctuation and lowercase ---
        removed_punctuation = ec.remove_punc(data["Data"])

        # --- Tokenise ---
        tokens = ec.tokeniser(removed_punctuation)

        # --- Remove stopwords ---
        tokened_removed_stopwords = ec.remove_stopwords(tokens)

        # --- Lemmatize ---
        lemmatized_words = ec.lemmatizer(tokened_removed_stopwords)

        vg_tokens = {"Name": data["Name"], "Tokens": lemmatized_words}
        document_tokens.append(vg_tokens)

    # print(document_tokens)
    # --- Get tf-idf scores for each doc based on the query terms ---
    doc_scores = ec.tfidf(query, document_tokens)

    # --- Get tf-idf score for the query ---
    query_scores = ec.query_tfidf(query, document_tokens)

    # --- Get cosine scores using vector normalisation and dot product ---
    dp_results = []
    for q in doc_scores:
        dp = ec.vector_space(query_scores["Vector"], q["Vector"])
        result_set = {"Name": q["Name"], "Dot product": dp}
        dp_results.append(result_set)

    # --- Cosine comparison (order & precision @ 10)
    ec.cosine_similarity(dp_results)


def main():
    q = enginex()
    engine1(q)

# main()