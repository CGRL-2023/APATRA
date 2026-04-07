#GPT-3.5 prediction as a target model
import pandas as pd
import openai
import re

# Configure LLM
openai.api_type = ""
openai.api_base = ""
openai.api_version = ""
openai.api_key = ""

def clean_word(word):
    return re.sub(r'[^\w\s]', '', word).strip()

def is_valid_word(word):
    return word.isalpha() and len(word) > 1

def get_mask_prediction(context):
    tokens = context.split()

    try:
        mask_index = tokens.index('<mask>') 
    except ValueError:
        raise ValueError("The context must contain a '<mask>' token.")

  
    context_with_mask = ' '.join(tokens)
    prompt = f"""
    This is the context: "{context_with_mask}"
    Please replace the '<mask>' with a single valid word that fits the sentence grammatically and contextually.
    """
    print(f"Prompt: {prompt}")

    predicted_word = None
   
    while predicted_word is None or not is_valid_word(predicted_word):
       
        response = openai.Completion.create(
            engine="",
            prompt=prompt,
            max_tokens=1,  
            temperature=0.2,
            top_p=0.05,
            frequency_penalty=0,
            presence_penalty=0
        )

        predicted_word = response.choices[0].text.strip()
        predicted_word = clean_word(predicted_word)

        if is_valid_word(predicted_word):
            break
        else:
            print(f"Invalid prediction received: '{predicted_word}'. Retrying...")

    return predicted_word

def create_adversarial_example(text):
    tokens = text.split()

    if '<mask>' not in tokens:
        raise ValueError("Input text must contain a '<mask>' token.")

    context = ' '.join(tokens)
    predicted_word = get_mask_prediction(context)

    print("Predicted token:", predicted_word)

    return predicted_word
#input path
input_csv = ''
df = pd.read_csv(input_csv)

results = []

for index, row in df.iterrows():
    masked_text = row['context']
    true_answer = row['answers']
    print(f"Processing row {index}: {masked_text}")
    try:
        predicted_token = create_adversarial_example(masked_text)
    except ValueError as e:
        print(f"Skipping row {index} due to error: {e}")
        continue
    except Exception as e:
        print(f"An unexpected error occurred for row {index}: {e}")
        continue

    results.append({
        'id': row['id'],
        'context': masked_text,  
        'true_answer': true_answer,
        'predicted_token': predicted_token  
    })


results_df = pd.DataFrame(results)
output_csv = ''
results_df.to_csv(output_csv, index=False)

print(f"Adversarial examples saved to {output_csv}")
