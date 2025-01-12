# import csv
# import engine_components as ec
#
# def read_CSV(file):
#     video_games = []
#     with open(file, 'r') as content:
#         for line in csv.DictReader(content): # Read each line and put it into a dictionary
#             video_games.append(line)
#
#     # Add name key with games name
#     for i in video_games:
#         name = i["url"].split('/')[-1]
#         name = name.replace('.html', '') # Remove .html
#         name = name.replace('-', ' ') # Replace - with whitespace
#         i["Game name"] = name # Add new field to dictionary
#
#     return video_games
#
# def CSV_search(vg, query, text_normalisation):
#     print("search")
#     results = []
#     terms = []
#
#     if text_normalisation == 1:
#         terms = ec.query_dealer(query, 1) # Lemmatization
#
#     elif text_normalisation == 2:
#         terms = ec.query_dealer(query, 2) # Stemming
#
#     elif text_normalisation == 3:
#         terms = ec.query_dealer(query, 3) # No lem/stem
#
#     print(terms)
#
#     for i in vg:
#         for j in terms.split(' '):
#             print()
#
#
# def main():
#     print("CSV")
#     video_games = read_CSV('videogame-labels.csv')
#
#     for i in video_games:
#         print(i)
#
#     CSV_search(video_games, 'Good and Bad?', 1)
#
# main()