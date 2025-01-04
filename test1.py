import csv
import regex
import requests
from bs4 import BeautifulSoup, Comment
import os

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
def test2(w):
    # URL = "https://en.wikipedia.org/wiki/Fernando_Alonso"
    URL = "https://en.wikipedia.org/wiki/Kelly_Rohrbach"
    a = "https://"
    # URL = "videogame/ps2.gamespy.com/24.html"
    # URL = a + "videogame/ps2.gamespy.com/24.html"
    print(URL)
    r = requests.get(URL) # gets html from URL
    # print(r.content)
    soup = BeautifulSoup(r.content, 'html5lib') # r.content = raw html, html5lib = html parser
    print(soup.prettify()) # formats html


def test3():
    # loop through all files in folder and print contents
    directory = r"C:\Users\Kimia\Documents\UEA\search_engine\videogames"

    # loop through videogames file

    for file in os.listdir(directory):
        with open(os.path.join(directory, file)) as f:
            print(f"content: {file}")
            # print(f.read()) # print contents of file
        print()

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
    s = open('videogames\\fifa-soccer-2005.html', "r")
    # print(s.read())
    a = "videogames\\24.html"
    # r = requests.get(a)
    # soup = BeautifulSoup(r.content, 'html5lib')
    soup = BeautifulSoup(s.read(), 'html5lib')
    # print(soup)

    # Remove css and js from scraped html
    for data in soup(['style', 'script']):
        data.decompose()

    # Get all text from under the content id
    data = soup.find_all(id='content')
    refined_data = str(data[0].get_text())
    pattern = regex.compile(r"\s+")
    simplified_data = regex.sub(pattern, " ", refined_data)
    print(simplified_data)


    '''
    1 - remove punctuation
        - regex
    2 - remove stopwords
        - nltk 
    3 - save bigrams
    4 - tokenise
    5 - stem/lem
        - prefer stem?
    '''

#Write to file



# (if needed)


def main():
    print("File Manager")
    # cvsReader()
    # test2("a")
    # test3()
    tes4()

main()