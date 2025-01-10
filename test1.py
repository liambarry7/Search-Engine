import csv
import time

import regex
import requests
from bs4 import BeautifulSoup, Comment
import os
import pandas

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from VideoGame import VideoGame


#Read file
# read .csv file
def cvsReader():
    # Reads csv file
    with open('videogame-labels.csv', mode = 'r') as file:
        myFile = csv.reader(file)
        myFile = csv.DictReader(file)



        line_count = 0
        list_of_games = []

        for line in myFile:
            if (line_count >= 1): # skips first line in csv file
                game_dict = {
                    "URL": line[0],
                    "Age Rating": line[1],
                    "Publisher": line[2],
                    "Genre": line[3],
                    "Developer": line[4],
                    "Description": ""
                }
                list_of_games.append(game_dict)

                """
                game = VideoGame(line[0], line[1], line[2], line[3], line[4])
                g = game.createDict()
                list_of_games.append(game.createDict())
                print(g)
                """

            line_count +=1
            # words = line.split(", ")
            # print(line)


    for i in list_of_games:
        print(i)


# Taking the html off a website
def test2():
    # URL = "https://en.wikipedia.org/wiki/Fernando_Alonso"
    URL = "https://en.wikipedia.org/wiki/Kelly_Rohrbach"
    a = "https://"
    URL = "videogame/ps2.gamespy.com/24.html"
    # URL = a + "videogame/ps2.gamespy.com/24.html"
    print(URL)
    r = requests.get(URL) # gets html from URL
    # print(r.content)
    soup = BeautifulSoup(r.content, 'html.parser') # r.content = raw html, html5lib = html parser
    # soup = BeautifulSoup(r.content, 'html5lib') # r.content = raw html, html5lib = html parser
    print(soup.prettify()) # formats html


def test3():
    # loop through all files in folder and print contents
    directory = r"C:\Users\Kimia\Documents\UEA\search_engine\videogames"

    # loop through videogames file

    for file in os.listdir(directory):
        time.sleep(10)
        with open(os.path.join(directory, file)) as f:
            # print(f"content: {file}")
            # print(f.read()) # print contents of file

            soup = BeautifulSoup(f.read(), 'html5lib')
            soup.prettify()
            print(soup)

            # Remove css and js from scraped html
            for data in soup(['style', 'script']):
                data.decompose()

            # Get all text from under the content id
            # data = soup.find_all(id='content')
            # refined_data = str(data[0].get_text())
            # pattern = regex.compile(r"\s+")
            # simplified_data = regex.sub(pattern, " ", refined_data)
            # print(simplified_data)

        print()

def fileOpener():
    file = os.listdir('videogames')
    print(file)
    file.remove(file[0]) # Remove .DS_store from list of files

    # Create file directory list
    file_list = []
    for i in range(len(file)):
        web_pages = "videogames/" + file[i]
        file_list.append(web_pages)

    return file_list

def scraperTest(paths):
     print("scraper2")
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

         print(simplified_data) # Return this

def remove_tags(html):

    # parse html content
    soup = BeautifulSoup(html, "html.parser")

    for data in soup(['style', 'script']):
        # Remove tags
        data.decompose()

    # return data by retrieving the tag content
    return ' '.join(soup.stripped_strings)

def tes4():
    # s = open('videogames\\24.html', "r")
    s = open('videogames\\hackgu.html', "r")
    # print(s.read())
    a = "videogames\\24.html"
    # r = requests.get(a)
    # soup = BeautifulSoup(r.content, 'html5lib')
    soup = BeautifulSoup(s.read(), 'html5lib')
    # print(soup)

    # Get game title for bigram
    raw_title = soup.find_all("title")
    game_title = regex.sub(r"GameSpy: ", "", raw_title[0].get_text())
    print(game_title)

    # Find
    # Remove css and js from scraped html
    for data in soup(['style', 'script']):
        data.decompose()

    # Get all text from under the content id
    data = soup.find_all(id='content')
    refined_data = str(data[0].get_text())
    pattern = regex.compile(r"\s+")
    simplified_data = regex.sub(pattern, " ", refined_data)
    print(simplified_data)

    bigrams = []
    bigrams.append(game_title)

    # remove punctuation and lowercase
    removed_punctation = regex.sub(r"[^\w\s]", "", simplified_data)
    removed_punctation = removed_punctation.lower()
    # print(removed_punctation)


    # Get stopwords
    nltk.download('stopwords')
    stopwords_list = set(stopwords.words('English'))
    # print(stopwords_list)

    # Tokenise
    nltk.download('punkt')
    tokens = word_tokenize(removed_punctation)

    tokenised_words = []
    for word in tokens:
        if word not in stopwords_list:
            tokenised_words.append(word)

    # print(tokenised_words)


    # Lemmatize
    nltk.download('wordnet')
    lm = WordNetLemmatizer()
    lemmatized_words = []

    for word in tokenised_words:
        lemmatized_words.append(lm.lemmatize(word))
    
    print(lemmatized_words)


    '''
    1 - remove punctuation
        - regex
    2 - remove stopwords
        - nltk 
    3 - save bigrams
        - game name?
        - publisher?
    4 - tokenise
    5 - stem/lem
        - prefer stem?
    '''

def test5():
    kgrams = []

    # get the name of html file from csv file
    videogame_details = pandas.read_csv('videogame-labels.csv')
    videogame_dict = videogame_details.to_dict(orient='records')  # Convert CVS file from pandas data structure to dictionary

    for record in videogame_dict:
        print(record)

    # get the url from each record reading to scrape, add to list
    list_of_urls = []
    for record in videogame_dict:
    #     if "a" in record.get('publisher'):
    #         kgrams.append(record.get('publisher'))
    #
    #     elif " " in record['developer']:
    #         kgrams.append(record['developer'])


        url = record['url']
        list_of_urls.append(url)

    print(list_of_urls)

    # beautiful soup
    for url in list_of_urls:
        # remove unneccessary parts of url for html file search
        pattern = regex.compile(r"^.*?/.*?/")  # match from start, any character until first /, then repeat until next one
        simpleURL = regex.sub(pattern, "", url)
        # print(simpleURL)

        # r = requests.get('videogames/' + simpleURL)
        r = open('videogames/' + simpleURL, 'r')
        soup = BeautifulSoup(r.read(), 'html5lib')
        soup.prettify()
        # print(soup)

        # Get game title for bigram
        raw_title = soup.find_all("title")
        game_title = regex.sub(r"GameSpy: ", "", raw_title[0].get_text())
        kgrams.append(game_title)
        print(game_title)

        # Remove css and js from scraped html
        for data in soup(['style', 'script']):
            data.decompose()

        # Get all text from under the content id
        data = soup.find_all(id='content')
        refined_data = str(data[0].get_text())
        pattern = regex.compile(r"\s+")
        simplified_data = regex.sub(pattern, " ", refined_data)
        print(simplified_data)

    # print(kgram)

def get_bigrams():
    print("x")

#Write to file



# (if needed)


def main():
    print("File Manager")
    # cvsReader()
    # test2("a")
    # test3()
    # tes4()
    # test5()
    # test2()
    a = fileOpener()
    print (a)
    scraperTest(a)



main()