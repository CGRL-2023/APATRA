import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline


tokenizer = AutoTokenizer.from_pretrained("mukund/privbert")
model = AutoModelForMaskedLM.from_pretrained("mukund/privbert")

fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

#input path
input_csv = ''
df = pd.read_csv(input_csv)

if 'context' not in df.columns or 'answers' not in df.columns:
    raise ValueError("The input CSV must contain columns named 'context' and 'answers'.")

results = []


for index, row in df.iterrows():
    masked_text = row['perturbed_context'] 
    fill_results = fill_mask(masked_text)

   
    for result in fill_results:
        try:
            predicted_token = result['token_str']
            score = result['score']
            results.append({
                'id': row['id'],
                'masked_text': masked_text,
                'true_answer': row['answers'],
                'predicted_token': predicted_token,
                'score': score
            })
        except (TypeError, KeyError) as e:
            print(f"Error processing result {result} for '{masked_text}': {e}")
            continue


results_df = pd.DataFrame(results)

output_csv = ''
results_df.to_csv(output_csv, index=False)

print(f"Results saved to {output_csv}")
