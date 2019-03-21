

from source.read_in_and_cleaning import load_data_and_clean, retrieve_specific_data_from_id, get_all_headlines

data = load_data_and_clean()

headlines = get_all_headlines(data)

print(headlines)
print(len(headlines[0]))
