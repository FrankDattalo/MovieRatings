import pandas as pd
import numpy as np

def read_to_dataframe(file_location):
    data = pd.read_csv(file_location, index_col=False)
    
    categorize_genres(data)

    categorize_directors(data)

    adjust_budget(data)

    # TODO: other preprocessing

    adjust_imdb_score(data)

    delete_extra_columns(data)

    return data

def adjust_imdb_score(data):
    max_imdb_score = 10

    # scales to a range between zero and one, while still preserving distances/variance
    data['imdb_score'] = data['imdb_score'] / max_imdb_score

def adjust_budget(data):

    # the average inflation rate over the last 100 years
    avg_inflation_rate = .0318
    
    # fill any data with missing values using means of columns
    data['budget'].fillna(data['budget'].mean(), inplace=True)
    data['title_year'].fillna(data['title_year'].mean(), inplace=True)

    # used for inflation calculation
    this_year = 2017

    # adjust for future value of movie (inflation)
    data['budget'] = data['budget'] * ((1 + avg_inflation_rate) ** (this_year - data['title_year']))
    
    # scale to unit stddev and 0 mean
    data['budget'] = (data['budget'] - data['budget'].mean()) / data['budget'].std()

def delete_extra_columns(data):
    del data['color']
    del data['num_critic_for_reviews']
    del data['director_facebook_likes']
    del data['actor_1_facebook_likes']
    del data['actor_2_facebook_likes']
    del data['actor_3_facebook_likes']
    del data['cast_total_facebook_likes']
    del data['movie_facebook_likes']
    del data['facenumber_in_poster']
    del data['plot_keywords']
    del data['movie_imdb_link']
    del data['num_user_for_reviews']
    del data['title_year']
    del data['aspect_ratio']
    del data['gross']
    del data['num_voted_users']


def categorize(data, list_of_categories, column_to_categorize):

    # create unknown category
    unknown_column = data[column_to_categorize].isin(list_of_categories) != True 
    data[column_to_categorize + '_is_unknown'] = unknown_column.astype(int)

    # other categories from list
    for category in list_of_categories:
        data[column_to_categorize + '_is_' + category] = (data[column_to_categorize] == category).astype(int)

    # remove original column now that it is categorized
    del data[column_to_categorize]


def categorize_genres(data):
    genres = ['Biography', 'Fantasy', 'Horror', 'Romance', 'Family', 
              'Sport', 'Mystery', 'Short', 'Music', 'Documentary', 
              'Sci-Fi', 'Crime', 'Drama', 'Thriller', 'Western', 'Comedy',
              'Musical', 'Action', 'Adventure', 'History', 'Animation', 'War']

    # data['genres'] is now a list of strings, for each genre, check if that genre is in the list
    data['genres'] = data['genres'].apply(lambda x: x.split('|'))

    data['genre_is_unknown'] = True

    for genre in genres:
        genre_is_x = 'genre_is_' + genre 
        data[genre_is_x] = data['genres'].apply(lambda arr: genre in arr)

        #things are unknown so long as they have not been found a genre
        data['genre_is_unknown'] = data.apply(lambda row: row['genre_is_unknown'] and not row[genre_is_x], axis=1)
        
        data[genre_is_x] = data[genre_is_x].astype(int)

    data['genre_is_unknown'] = data['genre_is_unknown'].astype(int)

    del data['genres']

def categorize_directors(data):

    # used to obtain director grouping
    g = data.groupby(['director_name'])
    
    # List of tuples of (director_name, count_of_movies) 
    director_to_movies = [(director, len(movies)) for (director, movies) in g.groups.items()]

    # sorts in descending order by count of movies
    director_to_movies.sort(key=lambda x: x[1], reverse=True)

    # top 50 directors by count of movies
    top_directors = [director for (director, _) in director_to_movies[0:50]]

    categorize(data, top_directors, 'director_name')

data = read_to_dataframe('movie_metadata.csv')