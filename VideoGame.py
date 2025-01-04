class VideoGame():
    def __init__(self, URL, age_rating, publisher, genre, developer):
        self.URL = URL
        # self.name = name
        self.age_rating = age_rating
        self.publisher = publisher
        self.genre = genre
        self.developer = developer

    def __str__(self):
        return self.URL, self.age_rating, self.publisher, self.genre, self.developer

    def createDict(self):
        gameDict = {
            "URL" : self.URL,
            "Age Rating" : self.age_rating,
            "Publisher" : self.publisher,
            "Genre" : self.genre,
            "Developer" : self.developer
        }

        return gameDict