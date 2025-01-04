import pandas
import regex
import requests
from bs4 import BeautifulSoup, Comment

def webScraper():
    # get the name of html file from csv file
    videogame_details = pandas.read_csv('videogame-labels.csv')
    videogame_dict = videogame_details.to_dict(orient='records') # Convert CVS file from pandas data structure to dictionary

    # for record in videogame_dict:
    #     print(record)

    # get the url from each record reading to scrape, add to list
    list_of_urls = []
    for record in videogame_dict:
        url = record['url']
        list_of_urls.append(url)

    print(list_of_urls)

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
        #
        # comments = soup.find_all(string=lambda text: isinstance(text, Comment))

        # Loop through each comment to find the comment that includes 'DESCRIPTION'
        # for comment in comments:
        #     if "DESCRIPTION" in comment:
        #
        #         # Extract all content between the comments
        #         next_sibling = comment.find_next_sibling()
        #         text_content = []  # List to contain all the text after the comment
        #
        #         while next_sibling and not isinstance(next_sibling, Comment):
        #             if next_sibling.name == 'p':
        #                 text_content.append(next_sibling.get_text())
        #             elif isinstance(next_sibling, str) and next_sibling.startswith('"') and next_sibling.endswith('"'):
        #                 text_content.append(next_sibling)
        #
        #             next_sibling = next_sibling.next_sibling
        #
        #         print(text_content)
        #
        #         game_info = ''.join(str(item) for item in text_content)  # Combine each of the strings in the list text_content into one string
        #         game_info.strip()  # Clean up the string by removing any leading or trailing whitespace characters (e.g. space, newline and tabs)
        #         # print(game_info)
        #         break  # Ignore any content after closing /DESCRIPTION comment

        # break





def getQuery():
    query = input("Please enter your query: ")
    return query

def main():
    # query = getQuery()
    # print("Your query is:", query)
    webScraper()

main()