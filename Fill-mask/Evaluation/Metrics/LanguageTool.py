
import pandas as pd
import language_tool_python


tool = language_tool_python.LanguageTool('en-US')


def check_grammar(text):
    matches = tool.check(text)
    num_errors = len(matches)
    score = max(0, 100 - num_errors)  
    return num_errors, score


input_csv = ''  # Update this with your file path
df = pd.read_csv(input_csv)


results = []
scores = []


for index, row in df.iterrows():
    context = row['updated_context']
    try:
        num_errors, score = check_grammar(context)
        results.append({
            'id': row['id'],  
            'context': context,
            'num_errors': num_errors,
            'grammar_score': score
        })
        scores.append(score)
    except Exception as e:
        print(f"An error occurred for row {index}: {e}")
        results.append({
            'id': row['id'], 
            'context': context,
            'num_errors': None,
            'grammar_score': None
        })


results_df = pd.DataFrame(results)


output_csv = ''
results_df.to_csv(output_csv, index=False)


if scores:
    overall_score = sum(scores) / len(scores)  
else:
    overall_score = None

print(f"Grammar check results saved to {output_csv}")
print(f"Overall Grammar Score: {overall_score}")

