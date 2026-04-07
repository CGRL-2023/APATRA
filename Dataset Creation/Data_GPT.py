import pandas as pd
import re
from openai import AzureOpenAI

# Replace with API
client = AzureOpenAI(
    api_key="",
    api_version="",
    azure_endpoint=""
)

def find_keyword_gpt4(context, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="", #add model
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Given the context below, identify the most significant single keyword highly related to the privacy policy "
                            f"(only return a single word containing only letters).\n\nContext: {context}\n\nKeyword:"
                        )
                    }
                ],
                max_tokens=10,
                temperature=0,
                seed=33
            )
            keyword = response.choices[0].message.content.strip()
            print(f"GPT-4 Response: {keyword}")
            if keyword.isalpha() and 3 <= len(keyword) <= 15:
                return keyword
        except Exception as e:
            print(f"Error: {e}")
    return None

def fallback_keyword(context):
    words = re.findall(r'\b\w+\b', context)
    return max(words, key=len) if words else ""

def replace_with_mask_and_update_answer(row):
    context = row['purified_context']
    keyword = find_keyword_gpt4(context)
    if keyword and keyword in context:
        start_index = context.find(keyword)
        masked_context = context.replace(keyword, '<mask>', 1)
    else:
        keyword = fallback_keyword(context)
        start_index = context.find(keyword) if keyword in context else -1
        masked_context = context.replace(keyword, '<mask>', 1) if keyword in context else context
    answer_str = f"{{'answer_start': [{start_index}], 'text': ['{keyword}']}}"
    return pd.Series([masked_context, answer_str])

#input path
df = pd.read_csv("")
df[['masked_context', 'answers']] = df.apply(replace_with_mask_and_update_answer, axis=1)

df.to_csv("masked_context_gpt4.csv", index=False)
