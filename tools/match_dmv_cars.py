import pandas as pd
import re
from fuzzywuzzy import fuzz, process
from core.common import cpath

def preprocess_column(column):
    """
    Preprocess a column by converting to lowercase, stripping whitespace, and removing special characters.
    """
    return column.str.lower().str.strip().str.replace(r'[^a-z0-9\s]', '', regex=True)

def match_dmv_to_cars(dmv, cars):
    """
    Match DMV makes and models to car companies and series.

    Parameters:
    - dmv: DataFrame containing DMV data with 'make' and 'model' columns.
    - cars: DataFrame containing car companies and series.

    Returns:
    - DataFrame with matched results.
    """
    # Preprocess columns for matching
    dmv['processed_make'] = preprocess_column(dmv['make'])
    dmv['processed_model'] = preprocess_column(dmv['model'])
    cars['processed_company'] = preprocess_column(cars['company'])
    cars['processed_serie'] = preprocess_column(cars['serie'])

    # Create combined columns for matching
    dmv['combined_dmv'] = dmv['processed_make'] + " " + dmv['processed_model']
    cars['combined_cars'] = cars['processed_company'] + " " + cars['processed_serie']

    # Ensure combined columns are lists of strings
    dmv_combined_list = dmv['combined_dmv'].astype(str).tolist()
    cars_combined_list = cars['combined_cars'].astype(str).tolist()

    # Perform fuzzy matching (optimized)
    matches = process.extract(dmv_combined_list, cars_combined_list, scorer=fuzz.token_sort_ratio, limit=1)
    matches = [{'dmv_entry': dmv_entry, 'best_match': match[0] if match else None, 'score': match[1] if match else None}
               for dmv_entry, match in zip(dmv_combined_list, matches)]

    # Convert matches to DataFrame
    matches_df = pd.DataFrame(matches)

    # Merge results back to original data for context
    result = (
        dmv
        .merge(matches_df, left_on='combined_dmv', right_on='dmv_entry', how='left')
        .merge(cars, left_on='best_match', right_on='combined_cars', how='left', suffixes=('_dmv', '_cars'))
    )

    return result[['make', 'model', 'company', 'serie', 'score']]

# Example usage
if __name__ == "__main__":
    # Load your dataframes here
    dmv = pd.read_csv(cpath('elv_data','dmv-augmented.csv.zip'))
    print('read dmv')
    cars = pd.read_csv(cpath('elv_data','cars-dataset.csv'))
    print('read cars')

    # Randomly select 100 rows from DMV data
    dmv_sample = dmv.sample(n=100, random_state=42)

    # Perform matching
    matched_results = match_dmv_to_cars(dmv_sample, cars)

    # Display results
    print(matched_results)
