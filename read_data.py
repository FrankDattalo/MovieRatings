import pandas as pd
import numpy as np
import sklearn as skl
import math


def read_to_dataframe(file_location):
    data = pd.read_csv(file_location, index_col=False)

    categorize_genres(data)
    categorize_directors(data)
    adjust_budget(data)
    adjust_duration(data)
    categorize_languages(data)
    categorize_rating(data)
    categorize_country(data)
    categorize_actor_1(data)
    categorize_actor_2(data)
    categorize_actor_3(data)
    #bucket_imdb_score(data)
    adjust_imdb_score(data)
    delete_extra_columns(data)

    return data


def adjust_imdb_score(data):
    max_imdb_score = 10

    # scales to a range between zero and one, while still preserving distances/variance
    data['imdb_score'] = data['imdb_score'] / max_imdb_score

# if we go with a catorizing model, this will need to be used
def bucket_imdb_score(data):
    def rount_to_nearest_half(score):
        return round(score / .5) * .5

    data['imdb_score'] = data['imdb_score'].apply(rount_to_nearest_half)

def adjust_budget(data):
    convert_to_usd(data)

    # the average inflation rate over the last 100 years
    avg_inflation_rate = .0318

    # fill data with missing values using means of columns
    data['title_year'].fillna(data['title_year'].mean(), inplace=True)

    # used for inflation calculation
    this_year = 2017

    # adjust for future value of movie (inflation)
    data['budget'] = data['budget'] * ((1 + avg_inflation_rate) ** (this_year - data['title_year']))

    # data['budget_after_inflation_adjustment'] = data['budget']

    # scale to unit stddev and 0 mean
    data['budget'] = (data['budget'] - data['budget'].mean()) / data['budget'].std()


def convert_to_usd(data):
    # usd / coversion currency
    conversion = {
        'USA': 1,  # usd
        'UK': .79,  # pounds
        'New Zealand': 1.49,  # nz dollars
        'Canada': 1.28,  # canadian dollar
        'Australia': 1.3,  # australian dollar
        'Belgium': .86,  # euro
        'Japan': 113.67,  # yen
        'Germany': .86,  # euro
        'China': 6.66,  # yuan
        'France': .86,  # euro
        'Mexico': 19.14,  # mexican peso
        'Spain': .86,  # euro
        'Hong Kong': 7.8,  # hong kong dollar
        'Czech Republic': 22.07,  # koruna
        'India': 64.89,  # rupee
        'South Korea': 1125.97,  # won
        'Italy': .86,  # euro
        'Russia': 58.03,  # ruble
        'Denmark': .16,  # krone
        'Ireland': .86,  # euro
        'South Africa': 14.14,  # rand
        'Iceland': 105.49,  # krona
        'Switzerland': 1,  # franc
        'Romania': 3.96,  # leu
        'Thailand': 33.23,  # bat
        'Iran': 34489,  # rhal
        'Poland': 3.66,  # zloty
        'Brazil': 3.24,  # real
        'Argentina': 17.6,  # argentine peso
        'Israel': 3.54  # new shekel
    }

    def update_currency(row):
        if row['country'] in conversion:
            return row['budget'] / conversion[row['country']]
        else:
            return (
            float('inf') / float('-inf'))  # return not a number, will fill missing countries with average budget

    data['budget'] = data.apply(update_currency, axis=1)

    # data['budget_after_conversion'] = data['budget']

    data['budget'].fillna(data['budget'].mean(), inplace=True)


def adjust_duration(data):
    # fill in empty values with mean
    data['duration'].fillna(data['duration'].mean(), inplace=True)

    # scale
    data['duration'] = (data['duration'] - data['duration'].mean()) / data['duration'].std()


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

        # things are unknown so long as they have not been found a genre
        data['genre_is_unknown'] = data.apply(lambda row: row['genre_is_unknown'] and not row[genre_is_x], axis=1)

        data[genre_is_x] = data[genre_is_x].astype(int)

    data['genre_is_unknown'] = data['genre_is_unknown'].astype(int)

    del data['genres']


def categorize_actor_1(data):
    a1 = data.groupby(['actor_1_name'])

    actor_to_movies = [(actor, len(movies)) for (actor, movies) in a1.groups.items()]
    actor_to_movies.sort(key=lambda x: x[1], reverse=True)

    top_actors = [actor for (actor, _) in actor_to_movies[0:50]]

    categorize(data, top_actors, 'actor_1_name')


def categorize_actor_2(data):
    a2 = data.groupby(['actor_2_name'])

    actor_to_movies = [(actor, len(movies)) for (actor, movies) in a2.groups.items()]
    actor_to_movies.sort(key=lambda x: x[1], reverse=True)

    top_actors = [actor for (actor, _) in actor_to_movies[0:50]]

    categorize(data, top_actors, 'actor_2_name')


def categorize_actor_3(data):
    a3 = data.groupby(['actor_3_name'])

    actor_to_movies = [(actor, len(movies)) for (actor, movies) in a3.groups.items()]
    actor_to_movies.sort(key=lambda x: x[1], reverse=True)

    top_actors = [actor for (actor, _) in actor_to_movies[0:50]]

    categorize(data, top_actors, 'actor_3_name')


# Use this if we want to group actor_1/2/3 together
def categorize_all_actors(data):
    # Get new column with all actors grouped together
    data['actor_name'] = pd.concat([data['actor_1_name'].dropna(), data['actor_2_name'].dropna(), data['actor_3_name']
                                   .dropna()]).reindex_like(data)

    a = data.groupby(['actor_name'])

    actor_to_movies = [(actor, len(movies)) for (actor, movies) in a.groups.items()]
    actor_to_movies.sort(key=lambda x: x[1], reverse=True)

    top_actors = [actor for (actor, _) in actor_to_movies[0:50]]

    categorize(data, top_actors, 'actor_name')


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


def categorize_country(data):
    countries = ['USA', 'UK', 'New Zealand', 'Canada', 'Australia', 'Belgium', 'Japan', 'Germany', 'China', 'France',
                 'Mexico', 'Spain'
        , 'Hong Kong', 'Czech Republic', 'India', 'Soviet Union', 'South Korea', 'Peru', 'Italy', 'Russia'
        , 'Aruba', 'Denmark', 'Libya', 'Ireland', 'South Africa', 'Iceland', 'Switzerland', 'Romania'
        , 'West Germany', 'Chile', 'Netherlands', 'Hungary', 'Panama', 'Greece', 'Sweden', 'Norway'
        , 'Taiwan', 'Cambodia', 'Thailand', 'Slovakia', 'Bulgaria', 'Iran', 'Poland', 'Georgia', 'Turkey'
        , 'Nigeria', 'Brazil', 'Finland', 'Bahamas', 'Argentina', 'Colombia', 'Israel', 'Egypt', 'Kyrgyzstan'
        , 'Indonesia', 'Pakistan', 'Slovenia', 'Afghanistan', 'Dominican Republic', 'Cameroon', 'United Arab Emirates'
        , 'Kenya', 'Philippines']

    categorize(data, countries, 'country')


def categorize_rating(data):
    ratings = ['PG-13', 'PG', 'G', 'R', 'NC-17', 'X', 'M', 'Unrated', 'GP', 'Approved']

    categorize(data, ratings, 'content_rating')


def categorize_languages(data):
    languages = ['Aboriginal', 'Arabic', 'Aramaic', 'Bosnian', 'Cantonese', 'Chinese', 'Czech',
                 'Danish', 'Dari', 'Dutch', 'Dzongkha', 'English', 'Filipino', 'French', 'German', 'Greek',
                 'Hebrew', 'Hindi', 'Hungarian', 'Icelandic', 'Indonesian', 'Italian', 'Japanese', 'Kannada',
                 'Kazakh', 'Korean', 'Mandarin', 'Maya', 'Mongolian', 'Norwegian', 'Panjabi', 'Persian',
                 'Polish', 'Portuguese', 'Romanian', 'Russian', 'Slovenian', 'Spanish', 'Swahili', 'Swedish', 'Tamil',
                 'Telugu', 'Thai', 'Urdu', 'Vietnamese', 'Zulu']

    categorize(data, languages, 'language')


def load_data(file_name='./movie_metadata_update.csv', train_percept=.75, reshape=False):
    dataframe = read_to_dataframe(file_name)

    split_index = math.floor(len(dataframe) * train_percept) 

    dataframe = skl.utils.shuffle(dataframe)

    y_train = dataframe[:split_index]['imdb_score']
    y_test  = dataframe[split_index:]['imdb_score']
    del dataframe['imdb_score']
    
    titles_train = [title for title in dataframe[:split_index]['movie_title']]
    titles_test  = [title for title in dataframe[split_index:]['movie_title']]
    del dataframe['movie_title']
    
    x_train = np.array(dataframe[:split_index])
    x_test  = np.array(dataframe[split_index:])

    y_train = np.array(y_train)
    y_test  = np.array(y_test)
    
    # some models expect thise to be a [x, 1] matrix while some expect them to
    # be 1D
    if reshape:
        y_train = y_train.reshape([x_train.shape[0], 1])
        y_test  = y_test.reshape([x_test.shape[0],  1])

    return (titles_train, x_train, y_train), (titles_test, x_test, y_test)
