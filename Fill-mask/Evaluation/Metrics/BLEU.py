import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu

nltk.download('punkt')


def replace_mask_with_answer(context, answer_text):
    return context.replace('<mask>', answer_text)


def extract_true_answer(answer):
    try:
        text_index = answer.find("'text'")     
        bracket_index = answer.find('[', text_index)     
        end_bracket_index = answer.find(']', bracket_index)
        text_value = answer[bracket_index + 1:end_bracket_index].strip("'\"[]")
        return text_value
    except Exception as e:
        print(f"Error extracting true answer: {e}")
        return None


def extract_predicted_token(prediction):
    try:
        predicted_token = row['predicted_token']
        return predicted_token
    except Exception as e:
        print(f"Error extracting predicted token: {e}")
        return None
#input path
input_csv = '' 
df = pd.read_csv(input_csv)

grouped = df.groupby('id')


bleu_scores_per_id = []


for group_id, group_data in grouped:
    all_context = []
    all_adversarial = []

   
    for index, row in group_data.iterrows():
        context = row['original_context']
        adversarial_context = row['masked_text']
        answer = row['true_answer']
        prediction = row['predicted_token']

        answer_text = extract_true_answer(answer)
        predicted_token = extract_predicted_token(prediction)

        if answer_text and predicted_token:
            updated_context = replace_mask_with_answer(context, answer_text)
            updated_adversarial_context = replace_mask_with_answer(adversarial_context, predicted_token)

            all_context.extend(nltk.word_tokenize(updated_context))
            all_adversarial.extend(nltk.word_tokenize(updated_adversarial_context))

    if all_context and all_adversarial:
        bleu_score = sentence_bleu([all_context], all_adversarial)
        bleu_scores_per_id.append(bleu_score)


overall_bleu_score = sum(bleu_scores_per_id) / len(bleu_scores_per_id) if bleu_scores_per_id else 0


print(f"BLEU scores per ID: {bleu_scores_per_id}")
print(f"Overall BLEU score: {overall_bleu_score}")