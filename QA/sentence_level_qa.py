import pandas as pd
import openai
from tqdm import tqdm


deployment_name = ''
openai.api_key = ""
openai.api_base = ""
openai.api_type = ''
openai.api_version = ''

def generate_new_sentence(context, question, original_answer):
    prompt = f"""
Given:
1. The original privacy policy context: {context}
2. The specific question being asked: {question}
3. The original answer to the question based on the context: {original_answer}

Create a new sentence to be added to the end of the privacy policy context by following these steps:

1. Identify the key information in the original context that answers the given question.

2. Extract a key phrase from the question that can be used to start a sentence. This phrase should be something that can naturally begin a statement in a privacy policy, such as "The types of information collected" or "We notify users".

3. Craft a new sentence that paraphrases the key information. Your new sentence should:
   - Start with the key phrase extracted from the question, adapting it as necessary to fit naturally into a privacy policy statement
   - Directly answer the given question
   - Convey a similar meaning to the original key information
   - Use different wording or sentence structure from the original context
   - Potentially introduce a subtle nuance or slight modification (e.g., adding or removing a minor detail)
   - Be relevant to the overall theme of the privacy policy
   - Be specific enough to draw attention, but not contradict the original information

4. Ensure that the new sentence:
   - Fits with the overall content of the privacy policy
   - Reads as part of a coherent privacy policy statement
   - Presents a subtle challenge for a QA model in relation to the specific question

Provide only the new sentence, without any explanations or additional text.
    """

    response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=[
            {"role": "system", "content": "You are an AI assistant that creates new sentences for privacy policies."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        temperature=0.7
    )

    return response['choices'][0]['message']['content']

def process_privacy_policy(context, question, original_answer):
    new_sentence = generate_new_sentence(context, question, original_answer)
    return context + " " + new_sentence

def process_file(input_file, output_file):
    df = pd.read_csv(input_file)
    df['modified_context'] = ''
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        context = row['context']
        question = row['question']
        original_answer = row['extracted_text']

        modified_context = process_privacy_policy(context, question, original_answer)

        df.at[index, 'modified_context'] = modified_context
    df.to_csv(output_file, index=False)
    print(f"Processing complete. Results saved to {output_file}")
input_file = 'input.csv'
output_file = 'output.csv'
process_file(input_file, output_file)