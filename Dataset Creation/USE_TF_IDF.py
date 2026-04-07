import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.feature_extraction.text import TfidfVectorizer
from google.colab import files


embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def get_embedding(text):
    return embed([text])[0].numpy()


def find_keyword(context):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([context])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    top_word = feature_array[tfidf_sorting[0]]
    return top_word

def replace_with_mask_and_update_answer(row):
    context = row['context']
    keyword = find_keyword(context)
    start_index = context.find(keyword)
    masked_context = context.replace(keyword, '<mask>', 1)
    answer = {'answer_start': [start_index], 'text': [keyword]}
    # Creating the string representation manually
    answer_str = f"{{'answer_start': [{start_index}], 'text': [{keyword}]}}"
    return pd.Series([masked_context, answer_str])

#input path
input_file = ''
df = pd.read_csv(input_file)


df[['masked_context', 'answers']] = df.apply(replace_with_mask_and_update_answer, axis=1)

#output path
output_file = ''
df.to_excel(output_file, index=False)


files.download(output_file)