import pandas as pd

def read_to_dataframe(file_location):
    data = pd.read_csv(file_location)
    # TODO: delete any unecessary columns or do any post processing 
    # necessary
    return data

def categorize(data_frame, list_of_categories, column_to_categorize):
    # TODO: implement categorize function
    return data_frame