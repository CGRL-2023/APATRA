
import pandas as pd
from sklearn.metrics import f1_score
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')


lemmatizer = WordNetLemmatizer()

#input path
csv_file_path = ''
df = pd.read_csv(csv_file_path)


df.columns = [ 'id','title','masked_text', 'true_answer', 'predicted_token', 'score']

# Extract the true answer text from the true_answer column
def extract_true_answer(answer):
    try:
        # Find the index of 'text' in the string
        text_index = answer.find("'text'")
        # Find the index of the first '[' after 'text'
        bracket_index = answer.find('[', text_index)
        # Find the index of the first ']' after '['
        end_bracket_index = answer.find(']', bracket_index)
        # Extract the text value between '[' and ']'
        text_value = answer[bracket_index+1:end_bracket_index].strip("'\"[]")
        return text_value
    except Exception as e:
        print(f"Error extracting true answer: {e}")
        return None


df['true_answer_text'] = df['true_answer'].apply(extract_true_answer)

print(df['true_answer_text'])


def lemmatize_text(text):
    text = str(text)  
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)



df['true_answer_text_lemmatized'] = df['true_answer_text'].apply(lemmatize_text)
df['predicted_token_lemmatized'] = df['predicted_token'].apply(lemmatize_text)

df.dropna(subset=['true_answer_text_lemmatized'], inplace=True)


max_score_rows = df.groupby('masked_text').apply(lambda x: x.loc[x['score'].idxmax()]).reset_index(drop=True)

# Calculate exact matches
exact_matches = (max_score_rows['true_answer_text_lemmatized'].str.lower() == max_score_rows['predicted_token_lemmatized'].str.lower()).sum()

# Calculate F1 score
f1 = f1_score(max_score_rows['true_answer_text_lemmatized'], max_score_rows['predicted_token_lemmatized'], average='macro')

print("Exact Matches:", exact_matches)
print("F1 Score:", f1)
