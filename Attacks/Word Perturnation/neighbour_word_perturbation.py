import pandas as pd
import re
from openai import AzureOpenAI

# APIs
client = AzureOpenAI(
    api_key="",
    api_version="",
    azure_endpoint=""
)

def clean_word(word):
    return re.sub(r'[^\w\s]', '', word).strip()

def get_replacement_word_gpt4(word, context):
    prompt = f"""Please provide a synonym for the word "{word}" that fits grammatically and contextually in the following sentence. 
    Ensure the synonym is not a special character or punctuation and is lowercase unless it's the first word of the sentence.\n\n{context}"""

    response = client.chat.completions.create(
        model="V1",  # Replace with your deployment name
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5
    )

    replacement_word = response.choices[0].message.content.strip()
    return clean_word(replacement_word) or word

def create_adversarial_example_gpt4(text):
    tokens = text.split()
    if '<mask>' not in tokens:
        raise ValueError("Input text must contain a '<mask>' token.")

    mask_index = tokens.index('<mask>')
    if mask_index > 0:
        tokens[mask_index - 1] = get_replacement_word_gpt4(tokens[mask_index - 1], text)
    if mask_index < len(tokens) - 1:
        tokens[mask_index + 1] = get_replacement_word_gpt4(tokens[mask_index + 1], text)

    return ' '.join(tokens)

#CSV file path
df = pd.read_csv("")

results = []
for index, row in df.iterrows():
    try:
        adv_text = create_adversarial_example_gpt4(row["context"])
        results.append({
            "id": row["id"],
            "context": row["context"],
            "answers": row["answers"],
            "adversarial_text": adv_text
        })
    except Exception as e:
        print(f"Error on row {index}: {e}")


pd.DataFrame(results).to_csv("", index=False)
