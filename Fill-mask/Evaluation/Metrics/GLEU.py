
import pandas as pd
import nltk
from nltk.translate.gleu_score import sentence_gleu

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

#input path
input_excel = ''  
df = pd.read_csv(input_excel)

individual_gleu_scores = []

results = []
for index, row in df.iterrows():
    context = row['context']
    adversarial_context = row['adversarial_context']
    answer = row['answers']
    answer_text = extract_true_answer(answer)

    if answer_text:
     
        updated_context = replace_mask_with_answer(adversarial_context, answer_text)
        context = replace_mask_with_answer(context, answer_text)


        original_tokens = nltk.word_tokenize(context)
        updated_tokens = nltk.word_tokenize(updated_context)

        gleu_score = sentence_gleu([original_tokens], updated_tokens)
        individual_gleu_scores.append(gleu_score)

    
        results.append({
            'id': row['id'],
            'context': context,
            'answers': row['answers'],
            'updated_context': updated_context,
            'gleu_score': gleu_score
        })
    else:
        print(f"Skipping row {index} due to error in extracting true answer.")


overall_gleu_score = sum(individual_gleu_scores) / len(individual_gleu_scores) if individual_gleu_scores else 0

results_df = pd.DataFrame(results)

output_excel = ''  
results_df.to_csv(output_excel, index=False)

print(f"Updated contexts and GLEU scores saved to {output_excel}")
print(f"Overall GLEU score: {overall_gleu_score}")

