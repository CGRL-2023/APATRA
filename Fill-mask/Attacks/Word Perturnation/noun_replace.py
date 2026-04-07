
import pandas as pd
import spacy
from openai import AzureOpenAI


nlp = spacy.load("en_core_web_sm")

#API
client = AzureOpenAI(
    api_key="YOUR_API_KEY",
    api_version="",
    azure_endpoint=""
)

# Function to replace object noun using GPT-4
def create_adversarial_example(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    mask_sentence = next((sent for sent in sentences if '<mask>' in sent.text), None)

    if not mask_sentence:
        raise ValueError("No <mask> token found in any sentence.")

    mask_sentence_text = mask_sentence.text

    noun_to_replace = None
    for token in mask_sentence:
        if token.pos_ == "NOUN" and token.dep_ in ["dobj", "obj", "iobj"] and token.text.lower() not in {"mask", "<", ">"}:
            noun_to_replace = token.text
            break

    if not noun_to_replace:
        raise ValueError("No suitable object noun found.")

    prompt = (
        f"Replace the noun '{noun_to_replace}' in the sentence below with a suitable synonym that preserves meaning and grammar. "
        f"Return only the full modified sentence.\n\nSentence: \"{mask_sentence_text}\""
    )

    response = client.chat.completions.create(
        model="",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=100,
    )

    updated_sentence = response.choices[0].message.content.strip()
    return text.replace(mask_sentence_text, updated_sentence)

# Load CSV
input_csv = ''
df = pd.read_csv(input_csv)

results = []
for index, row in df.iterrows():
    try:
        new_text = create_adversarial_example(row['context'])
    except Exception as e:
        print(f"Row {index} skipped due to error: {e}")
        continue

    results.append({
        'id': row['id'],
        'context': row['context'],
        'answers': row['answers'],
        'adversarial_text': new_text
    })

# Save results
results_df = pd.DataFrame(results)
output_csv = ''
results_df.to_csv(output_csv, index=False)
print(f"Saved to {output_csv}")
