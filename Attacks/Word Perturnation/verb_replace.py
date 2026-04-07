
# python -m spacy download en_core_web_sm

import pandas as pd
import nltk
import spacy
import re
from openai import AzureOpenAI

# Load resources
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")

# API
client = AzureOpenAI(
    api_key="",
    api_version="",
    azure_endpoint=""
)


def create_adversarial_example(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    mask_sentence = next((sent for sent in sentences if '<mask>' in sent.text), None)

    if not mask_sentence:
        raise ValueError("Input text must contain a '<mask>' token in one of the sentences.")

    mask_sentence_text = mask_sentence.text
    print(f"Original sentence: {mask_sentence_text}")

    verb_to_replace = None
    for token in mask_sentence:
        if token.pos_ == "VERB" and token.text.lower() not in {"do", "mask", "<", ">"}:
            verb_to_replace = token.text
            break

    if not verb_to_replace:
        raise ValueError("No suitable verb found for replacement.")

    print(f"Identified verb: {verb_to_replace}")

 
    prompt = (
        f"Replace the verb '{verb_to_replace}' in the following sentence with a suitable synonym "
        f"that fits grammatically and contextually. Return the full updated sentence only:\n\n"
        f"Sentence: \"{mask_sentence_text}\""
    )

    response = client.chat.completions.create(
        model="",  # GPT-4 deployment name
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=100,
    )

    adversarial_sentence = response.choices[0].message.content.strip()
    print(f"Adversarial sentence: {adversarial_sentence}")

   
    adversarial_text = text.replace(mask_sentence_text, adversarial_sentence)
    return adversarial_text

#input path
input_csv = ''
df = pd.read_csv(input_csv)

results = []

# Generate adversarial examples
for index, row in df.iterrows():
    masked_text = row['context']
    true_answer = row['answers']
    print(f"\nProcessing row {index}:\n{masked_text}")

    try:
        adversarial_text = create_adversarial_example(masked_text)
    except ValueError as e:
        print(f"Skipping row {index} due to error: {e}")
        continue
    except Exception as e:
        print(f"Unexpected error at row {index}: {e}")
        continue

    results.append({
        'id': row['id'],
        'context': masked_text,
        'answers': true_answer,
        'adversarial_text': adversarial_text
    })


results_df = pd.DataFrame(results)
output_csv = ''
results_df.to_csv(output_csv, index=False)
print(f"\nAdversarial examples saved to {output_csv}")
