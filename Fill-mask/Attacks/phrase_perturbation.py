
import pandas as pd
import re
from openai import AzureOpenAI
#API
client = AzureOpenAI(
    api_key="",
    api_version="",
    azure_endpoint="",
)


#input path
data = pd.read_csv("")

def generate_adversarial_sample_gpt(context):
  PROMPT_MESSAGES = [
    {
      "role": "user",
      "content": f"""
    You are strongly familiar with policy concepts and also an expert NLP assistant tasked with generating phrase-level adversarial examples for a given policy context.
    Follow these instructions carefully:

    1. Focus on sentences containing the <mask> token.
    2. Identify phrases (e.g., noun phrases, verb phrases, adjective phrases) within the sentence that can be replaced.
    3. Replace the identified phrase with a synonymous phrase or rephrased version.
       - Ensure the replacement does not change the meaning significantly.
       - Maintain grammatical correctness.
       - Make sure the replaced word must not be - any unmeaningful word, empty space, special symbol, punctuation, number
    4. If no replaceable phrases are found, replace single adjectives, verbs, or nouns with suitable synonyms.
    5. Preserve the <mask> tokens, first words, and last words of the sentence unchanged.

    Example:
    Input: Where we combine Other Information with Personal Information in a way that can identify you or be used to identify you personally, we will treat that information as Personal Information, subject to the <mask> of this Privacy Policy.
    Output: Where we merge Other Information with Personal Data in a manner that can reveal your identity or be used for identification, we will treat that data as Personal Information, subject to the <mask> of this Privacy Policy.

    Input Context:
    {context}

    Return only the modified context.
    """

    }
  ]

  try:
        params = {
        "model": "",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 500,
        "seed": 33,
        "temperature": 0,
    }
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content

  except Exception as e:
        print(f"Error generating adversarial sample: {e}")
        return context


data['adversarial_context'] = data['context'].apply(generate_adversarial_sample_gpt)

output_path = ""
data.to_csv(output_path, index=False)

print(f"Adversarial samples generated and saved to {output_path}.")
