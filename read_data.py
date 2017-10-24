import pandas as pd

def read_to_dataframe(file_location):
    data = pd.read_csv(file_location, index_col=False)
    
    delete_extra_columns(data)

    categorize_genres(data)

    categorize_directors(data)

    # TODO: delete any unnecessary columns or do any post processing 
    # necessary
    return data


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
    del data['content_rating']
    del data['title_year']
    del data['aspect_ratio']


def categorize(data, list_of_categories, column_to_categorize):
    """
    This function is used to categorize a a given column of the data from a list of catories
    """
    unknown_column = data[column_to_categorize].isin(list_of_categories) != True 
    data[column_to_categorize + '_is_unknown'] = unknown_column.astype(int)

    for category in list_of_categories:
        data[column_to_categorize + '_is_' + category] = (data[column_to_categorize] == category).astype(int)

    del data[column_to_categorize]


def categorize_genres(data):
    genres = ['Biography', 'Fantasy', 'Game-Show', 'Horror', 'Romance', 'Family', 
              'Sport', 'Mystery', 'Short', 'Reality-TV', 'Music', 'Documentary', 
              'Sci-Fi', 'News', 'Crime', 'Drama', 'Thriller', 'Western', 'Comedy',
              'Musical', 'Action', 'Adventure', 'History', 'Film-Noir', 'Animation', 'War']

    data['genres'] = data['genres'].apply(lambda x: x.split('|'))

    for genre in genres:
        data['genre_is_' + genre] = (data['genres'] == genre).astype(int)

    del data['genres']

def categorize_directors(data):

    # used to obtain director grouping
    g = data.groupby(['director_name'])
    
    # List of tuples of (director_name, count_of_movies) 
    director_to_movies = [(director, len(movies)) for (director, movies) in g.groups.items()]
    director_to_movies.sort(key=lambda x: x[1], reverse=True)

    # top 200 directors
    top_directors = [director for (director, _) in director_to_movies[0:200]]

    categorize(data, top_directors, 'director_name')