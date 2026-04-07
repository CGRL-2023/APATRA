
import pandas as pd
import openai
import re

# Set OpenAI API details
openai.api_type = ""
openai.api_base = ""  # Base URL of OpenAI service
openai.api_version = ""  # OpenAI version
openai.api_key = ""  # OpenAI API key

def clean_text(text):
  
    text = re.sub(r'[^\w\s]', '', text).strip()
    text = re.sub(r'\b(Task|Example|Prompt|Fill in the mask token|Based on the sentence|new_sentence)\b', '', text, flags=re.IGNORECASE)
    return text.strip()

# Function to get a relevant sentence from GPT
def get_relevant_sentence(masked_sentence):
    prompt = f"""You are an assistant specialized in writing privacy policies. Based on the sentence that contains a <mask> token, add one relevant and paraphrased version of sentence before it that is related to privacy policies.
    The new sentence should be simple, coherent, contextually relevant, and fit naturally with the original privacy policy content.
    The new sentence should introduce a new idea related to privacy and should not repeat or rephrase any part of the original sentence.
    The new sentence shouldn't contain any special characters, numbers, or symbols.
    Avoid any irrelevant information, tasks, numbers, or examples. Also, the new sentence must be grammatically correct.
    Original Sentence: {masked_sentence}

    New Sentence:"""


    print(f"Prompt: {prompt}")  

    response = openai.Completion.create(
        engine="",
        prompt=prompt,
        max_tokens=50,  
        temperature=0.2,
        top_p=0.05,
        frequency_penalty=0,
        presence_penalty=0
    )

    new_sentence = response.choices[0].text.strip()
    clean_sentence = clean_text(new_sentence)

    # Ensure the sentence is relevant and not empty after cleaning
    if not clean_sentence or len(clean_sentence.split()) < 5:  
        raise ValueError("Generated sentence was either irrelevant or too short.")

    return clean_sentence


def create_adversarial_example(text):
    sentences = text.split('. ')

   
    mask_sentence_index = None
    for i, sentence in enumerate(sentences):
        if '<mask>' in sentence:
            mask_sentence_index = i
            break

    if mask_sentence_index is None:
        raise ValueError("Input text must contain a '<mask>' token.")

    mask_sentence = sentences[mask_sentence_index]

    new_sentence = get_relevant_sentence(mask_sentence)


    sentences.insert(mask_sentence_index, new_sentence)
 
    adversarial_text = '. '.join(sentences).strip() + '.'
    print("Adversarial output:", adversarial_text)

    return adversarial_text
#input path
input_csv = ''
df = pd.read_csv(input_csv)

results = []


for index, row in df.iterrows():
    masked_text = row['context']
    true_answer = row['answers']
    print(f"Processing row {index}: {masked_text}")  
    try:
        adversarial_text = create_adversarial_example(masked_text)
    except ValueError as e:
        print(f"Skipping row {index} due to error: {e}")
        continue
    except Exception as e:
        print(f"An unexpected error occurred for row {index}: {e}")
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

print(f"Adversarial examples saved to {output_csv}")

