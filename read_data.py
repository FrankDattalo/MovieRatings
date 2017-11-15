import pandas as pd
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
import math
import time
import os

# This function is responsible for loading and processing the file into a 
# pandas data frame
def read_to_dataframe(file_location):
    data = pd.read_csv(file_location, index_col=False).dropna()
    
    age_to_see_movie(data)
    data = historical_data(data)
    movie_counts(data)
    
    inflation_adjustments(data)
    adjust_duration(data)

    categorize_genres(data)
    categorize_languages(data)
    categorize_country(data)

    # #categorize_rating(data)
    # #categorize_directors(data)
    # #categorize_actor_1(data)
    # #categorize_actor_2(data)
    # #categorize_actor_3(data)

    normalize_other_values(data)

    delete_extra_columns(data)
    
    return data

def normalize_other_values(data):
    historical_cols = ['imdb_score', 'gross', 'budget', 'title_year']
    person_names = ['actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name']
    stats = ['_past_mean_', '_past_median_', '_past_max_', '_past_min_']

    cols_to_normalize = ['title_year', 'age_to_see_movie', 'budget', 
                         'director_name_previous_movie_count', 'actor_1_name_previous_movie_count', 
                         'actor_2_name_previous_movie_count', 'actor_3_name_previous_movie_count']

    for historical_col in historical_cols:
        for person_name in person_names:
            for stat in stats:
                cols_to_normalize.append(person_name + stat + historical_col)

    for col in cols_to_normalize:
        zero_mean_unit_std(data, col)

def zero_mean_unit_std(data, col):
    data[col] = (data[col] - data[col].mean()) / data[col].std()

def age_to_see_movie(data):
    ratings = {'R': 17, 'PG-13': 13, 'PG': 10, 'G': 0, 'Not Rated': 18, 
               'Unrated': 18, 'Approved': 0, 'X': 17, 'NC-17': 17, 
               'Passed': 10, 'M': 17, 'GP': 10}

    data['age_to_see_movie'] = data['content_rating'].apply(lambda rating: ratings[rating])

def movie_counts(data):
    start = time.time()

    data_columns = ['movie_title', 'title_year']

    temp1 = data[['actor_1_name', *data_columns]]
    temp1.columns = ['person_name', *data_columns]

    temp2 = data[['actor_2_name', *data_columns]]
    temp2.columns = ['person_name', *data_columns]

    temp3 = data[['actor_3_name', *data_columns]]
    temp3.columns = ['person_name', *data_columns]

    # data fraome of all actors, combining actors 1, 2, and 3
    all_actors = pd.concat([temp1, temp2, temp3], axis=0)

    all_directors = data[['director_name', *data_columns]]
    all_directors.columns = ['person_name', *data_columns]

    i = 0

    # This code expressed as SQL:
    #
    # SELECT COUNT(*)
    # FROM MOVIES
    # WHERE TITLE_YEAR <= :TITLE_YEAR
    # AND   DIRECTOR_NAME = :DIRECTOR_NAME
    # AND   MOVIE_TITLE != :MOVIE_TITLE
    # 
    # SELECT COUNT(*)
    # FROM MOViES
    # WHERE TITLE_YEAR <= :TITLE_YEAR
    # AND   ACTOR_NAME = :ACTOR_NAME
    # AND   MOVIE_TITLE != :MOVIE_TITLE
    def counting_fn(row, director_or_actor_name):
        nonlocal i

        if i % 400 == 0:
            print("movie_counts() - Rows mapped", i // 4)

        i += 1

        if director_or_actor_name != 'director_name':
            past = all_actors
        else:
            past = all_directors

        past = past[past['title_year']  <= row['title_year']]
        past = past[past['movie_title'] != row['movie_title']]
        past = past[past['person_name'] == row[director_or_actor_name]]

        return len(past)

    for person_of_interest in ['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name']:
        data[person_of_interest + '_previous_movie_count'] = \
            data.apply(lambda row: counting_fn(row, person_of_interest), axis=1)

    end = time.time()

    print("movie_counts() - Total elapsed time in seconds:", (end - start))

def historical_data(data):
    # This function will assign the historical averages, mins, maxs, and medians
    # of the following historical columns for each movie
    # where person in {director, actor 1-3} in movie_x 
    # and movie_x.title_year <= this_movie.title_year 

    # btw, it also takes hella long to run (almost 3 minutes on my computer)

    start = time.time()

    historical_columns = ['imdb_score', 'gross', 'budget', 'title_year']
    temp_columns = ['movie_title', *historical_columns]

    # * means rest of arguments in python, unrolls list

    temp1 = data[['actor_1_name', *temp_columns]]
    temp1.columns = ['actor_name', *temp_columns]

    temp2 = data[['actor_2_name', *temp_columns]]
    temp2.columns = ['actor_name', *temp_columns]

    temp3 = data[['actor_3_name', *temp_columns]]
    temp3.columns = ['actor_name', *temp_columns]

    # data fraome of all actors, combining actors 1, 2, and 3
    all_actors = pd.concat([temp1, temp2, temp3], axis=0)

    # used for printing
    i = 0

    # This code as SQL
    # SELECT MIN(BUDGET), MAX(BUDGET), MEDIAN(BUDGET), MEAN(BUDGET), etc..
    # FROM MOVIES
    # WHERE TITLE_YEAR <= :TITLE_YEAR
    # AND   MOVIE_TITLE != :MOVIE_TITLE
    # AND   DIRECTOR_NAME = :DIRECTOR_NAME
    # (note this code also maps values for actors and will use averages of all
    #  if none for that specific actor or director is found)
    def historical_fn(row):
        nonlocal i
        if i % 100 == 0:
            print("historical_data() - Rows mapped", i)
        i += 1

        stats = {}

        past = data[data['title_year'] <= row['title_year']]
        # don't include the one we're trying to predict
        past = past[past['movie_title'] != row['movie_title']]
        director_stats = past[past['director_name'] == row['director_name']]

        for col_name in historical_columns:
            # if there are no historical statistics for this director up until this point
            if len(director_stats) == 0:
                # use the avaerage historical data instead
                stats['director_name_past_mean_' + col_name] = past[col_name].mean()
                stats['director_name_past_median_' + col_name] = past[col_name].median()
                stats['director_name_past_max_' + col_name] = past[col_name].max()
                stats['director_name_past_min_' + col_name] = past[col_name].min()
                continue

            stats['director_name_past_mean_' + col_name] = director_stats[col_name].mean()
            stats['director_name_past_median_' + col_name] = director_stats[col_name].median()
            stats['director_name_past_max_' + col_name] = director_stats[col_name].max()
            stats['director_name_past_min_' + col_name] = director_stats[col_name].min()
            
        past = all_actors[all_actors['title_year'] <= row['title_year']]
        # don't include the one we're trying to predict
        past = past[past['movie_title'] != row['movie_title']]
        
        for actor_x_name in ['actor_1_name', 'actor_2_name', 'actor_3_name']:
            actor_stats = past[past['actor_name'] == row[actor_x_name]]

            for col_name in historical_columns:
                # use the overall statistics up until this point
                if len(actor_stats) == 0:
                    # use the average historical data instead
                    stats[actor_x_name + '_past_mean_' + col_name] = past[col_name].mean()
                    stats[actor_x_name + '_past_median_' + col_name] = past[col_name].median()
                    stats[actor_x_name + '_past_max_' + col_name] = past[col_name].max()
                    stats[actor_x_name + '_past_min_' + col_name] = past[col_name].min()
                    continue
                    
                stats[actor_x_name + '_past_mean_' + col_name] = actor_stats[col_name].mean()
                stats[actor_x_name + '_past_median_' + col_name] = actor_stats[col_name].median()
                stats[actor_x_name + '_past_max_' + col_name] = actor_stats[col_name].max()
                stats[actor_x_name + '_past_min_' + col_name] = actor_stats[col_name].min()
                
        return stats

    # used as a temp column, which will store a map of variables.
    # It's currently done this way in order to speed up performance.
    # Rather than calling a function to collect statistics for each column individually,
    # They are all done in historical_fn and then parsed out in the loop below
    data['stats'] = data.apply(historical_fn, axis=1)

    # split the stats map into it's respective columns by grabbing what is needed
    # from the map at each row
    for person in ['actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name']:
        for historical_column in historical_columns:
            for _past_x_ in ['_past_mean_', '_past_median_', '_past_max_', '_past_min_']:
                full_column_name = person + _past_x_ + historical_column
                data[full_column_name] = data.apply(lambda row: row['stats'][full_column_name], axis=1)

    # remove the temporary column
    del data['stats']

    before = len(data)
    data = data.dropna()
    after = len(data)
    # the difference should only be 1. 
    # (the only movie that has no previous average and 
    #  no previous historical data is the oldest movie in the database)
    print("historical_data() - Dropping Na. Size before", before, "Size after", after)

    end = time.time()

    print("historical_data() - Total elapsed time in seconds:", round(end - start, 2))

    return data

def inflation_adjustments(data):
    data['title_year'].fillna(data['title_year'].mean(), inplace=True)

    cols_to_adjust = ['budget']

    historical_cols = ['gross', 'budget']
    person_names = ['actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name']
    stats = ['_past_mean_', '_past_median_', '_past_max_', '_past_min_']

    for historical_col in historical_cols:
        for person_name in person_names:
            for stat in stats:
                cols_to_adjust.append(person_name + stat + historical_col)

    for col in cols_to_adjust:
        adjust_for_inflation(data, col)

def adjust_for_inflation(data, col_name):
    convert_to_usd(data)

    # the average inflation rate over the last 100 years
    avg_inflation_rate = .0318

    # used for inflation calculation
    this_year = 2017

    # adjust for future value of movie (inflation)
    data[col_name] = data[col_name] * ((1 + avg_inflation_rate) ** (this_year - data['title_year']))

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
            return np.nan  # return not a number, will fill missing countries with average budget

    data['budget'] = data.apply(update_currency, axis=1)

    # data['budget_after_conversion'] = data['budget']

    data['budget'].fillna(data['budget'].mean(), inplace=True)


def adjust_duration(data):
    # fill in empty values with mean
    data['duration'].fillna(data['duration'].mean(), inplace=True)

    zero_mean_unit_std(data, 'duration')


def delete_extra_columns(data):
    cols_to_delete = ['color', 'num_critic_for_reviews', 'director_facebook_likes', 
                      'actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes',
                      'cast_total_facebook_likes', 'movie_facebook_likes', 'facenumber_in_poster',
                      'plot_keywords', 'movie_imdb_link', 'num_user_for_reviews',
                      'aspect_ratio', 'gross', 'num_voted_users',
                      'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name',
                      'content_rating']

    for col in cols_to_delete:
        del data[col]


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

    for genre in genres:
        genre_is_x = 'genre_is_' + genre
        data[genre_is_x] = data['genres'].apply(lambda arr: genre in arr).astype(int)

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
    director_to_movies = {director: len(movies) for (director, movies) in g.groups.items()}

    data['director_count'] = data['director_name'].apply(lambda director: director_to_movies[director])


def categorize_country(data):
    countries = ['USA', 'UK', 'France', 'Germany', 'Canada', 'Australia', 
                 'Spain', 'Japan', 'China', 'Hong Kong', 'New Zealand', 'Italy']

    categorize(data, countries, 'country')


def categorize_rating(data):
    ratings = ['PG-13', 'PG', 'G', 'R', 'NC-17', 'X', 'M', 'Unrated', 'GP', 'Approved']

    categorize(data, ratings, 'content_rating')


def categorize_languages(data):
    languages = ['English', 'French', 'Spanish', 'Mandarin', 
                 'German', 'Japanese', 'Cantonese', 'Italian']

    categorize(data, languages, 'language')


def load_data(file_name='./movie_metadata_update.csv', train_percept=.75, reshape=False, cache=True, process=True):
    cache_save_location = file_name + '.pkl'
    
    if cache and os.path.exists(cache_save_location):
        print('load_data() - Loading cached version.')
        dataframe = pd.read_pickle(cache_save_location)
    else:
        print('load_data() - Computing dataframe.')
        dataframe = read_to_dataframe(file_name)
        if cache:
            print('load_data() - Caching dataframe.')
            dataframe.to_pickle(cache_save_location)

    if not process:
        return dataframe

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


def plot_histogram(data_location='./movie_metadata_update.csv'):
    data_frame = pd.read_csv(data_location, index_col=False).dropna()
    plt.hist(data_frame['imdb_score'])
    plt.title('IMDB Score Distribution')
    plt.show()