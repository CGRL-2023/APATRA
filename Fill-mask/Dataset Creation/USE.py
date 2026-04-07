#dataset creation using USE only
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from google.colab import files

# Load the Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def get_embedding(text):
    return embed([text])[0].numpy()

def find_keyword_using_use(context):
    words = context.split()
    context_embedding = get_embedding(context)
    max_similarity = -1
    keyword = None

    for word in words:
        word_embedding = get_embedding(word)
        similarity = np.dot(context_embedding, word_embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            keyword = word

    return keyword

def replace_with_mask_and_update_answer(row):
    context = row['context']
    keyword = find_keyword_using_use(context)
    start_index = context.find(keyword)
    masked_context = context.replace(keyword, '<mask>', 1)
    answer = {'answer_start': [start_index], 'text': [keyword]}
    answer_str = f"{{'answer_start': [{start_index}], 'text': ['{keyword}']}}"
    return pd.Series([masked_context, answer_str])

#input path
input_file = ''
df = pd.read_csv(input_file)


df[['masked_context', 'answers']] = df.apply(replace_with_mask_and_update_answer, axis=1)

#output path
output_file = ''
df.to_csv(output_file, index=False)


files.download(output_file)
