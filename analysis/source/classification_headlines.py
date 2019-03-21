
import tensorflow as tf
from read_in_and_cleaning import load_data_and_clean, get_all_labels, get_all_headlines


data = load_data_and_clean()
headlines = get_all_headlines(data)
labels = get_all_labels(data)

print(headlines[:5])
print(labels[:5])

