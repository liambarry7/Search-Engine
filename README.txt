--- Video Game Search Engine ---

-- Description --
This program allows the user to search a dataset of 400 websites that contain content about PlayStation 2 video games. It features a multiple different search engine to demonstrate the effect and usefulness of different information retrieval techniques and processes. By using TF-IDF ranking and Cosine similarity, the program produces a result set at ten precision for all queries across all 7 search engines.

The program utilised the natural language toolkit to manipulate information scraped off each website to create a list of tokenised words than is compared to the search query. Depending on the similarity to the TF-IDF of the query, vector normalisation and dot product is used to create a cosine similarity score which is then used to rank each document against the query. The scores are ordered, and the top ten results are displayed to the user. Having seen the displayed results, the user can opt to choose another (or the same) search engine or exit the program.

-- Installation --
To run this program, the user will have to have installed the nltk, NumPy, spacy, regex, os, time, pandas and beautifulsoup. Following this, the user will also need the following datasets downloaded from nltk: 'stopwords', 'wordnet' and 'punkt'.

-- Usage --
In order to run the program, the user must run 'search_engines.py'. This starts the program off by downloading the necessary datasets from nltk before taking you to a main menu. Here the user can see all engines available for them to use and will be prompted to input the number of their desired engine. Next, the user will enter their query, which will be processed and will return a ranked result set at ten precision

-- Authors --
Liam Barry