import pandas as pd
import random
import string
import spacy
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


nlp = spacy.load('en_core_web_sm')
model_name = " "  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
model.eval()

def perturb_word(word, method):
    perturbations = []
    for i in range(len(word)):
        if method == 'replace':
            new_char = random.choice(string.ascii_letters)
            new_word = word[:i] + new_char + word[i + 1:]
        elif method == 'swap' and i < len(word) - 1:
            new_word = word[:i] + word[i+1] + word[i] + word[i+2:]
        elif method == 'delete':
            new_word = word[:i] + word[i+1:]
        elif method == 'insert':
            new_char = random.choice(string.ascii_letters)
            new_word = word[:i] + new_char + word[i:]
        elif method == 'repeat':
            new_word = word[:i] + word[i] + word[i:]
        elif method == 'keyboard':
            keyboard = {
                'q': 'wa', 'w': 'qe', 'e': 'wr', 'r': 'et', 't': 'ry', 'y': 'tu', 'u': 'yi', 'i': 'uo', 'o': 'ip', 'p': 'o[',
                'a': 'sz', 's': 'ad', 'd': 'sf', 'f': 'dg', 'g': 'fh', 'h': 'gj', 'j': 'hk', 'k': 'jl', 'l': 'k;',
                'z': 'x', 'x': 'zc', 'c': 'xv', 'v': 'cb', 'b': 'vn', 'n': 'bm', 'm': 'n,'
            }
            if word[i].lower() in keyboard:
                new_char = random.choice(keyboard[word[i].lower()])
                new_word = word[:i] + new_char + word[i+1:]
            else:
                new_word = word
        else:
            new_word = word
        
        if new_word != word: 
            perturbations.append(new_word)
    
    return perturbations

def find_answer_in_context(context, answer):
    context_lower = context.lower()
    answer_lower = answer.lower()
    
    start_idx = context_lower.find(answer_lower)
    if start_idx == -1:
        return None, None
    
    end_idx = start_idx + len(answer)
    return start_idx, end_idx

def get_qa_prediction(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    
    answer_tokens = inputs.input_ids[0][answer_start:answer_end]
    predicted_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    return predicted_answer

def process_qa_sample(policy_excerpt, question, answer, method):
    start_idx, end_idx = find_answer_in_context(policy_excerpt, answer)
    
    if start_idx is None:
        print(f"Warning: Answer '{answer}' not found in policy excerpt")
        return []

    answer_text = policy_excerpt[start_idx:end_idx]
    answer_words = answer_text.split()
    
    perturbed_samples = []
    for word_idx, word in enumerate(answer_words):
        perturbations = perturb_word(word, method)
        
        for perturbed_word in perturbations:
            perturbed_answer_words = answer_words.copy()
            perturbed_answer_words[word_idx] = perturbed_word
            perturbed_answer_text = ' '.join(perturbed_answer_words)
            perturbed_policy = (
                policy_excerpt[:start_idx] + 
                perturbed_answer_text + 
                policy_excerpt[end_idx:]
            )

            predicted_answer = get_qa_prediction(question, perturbed_policy)
            
            perturbed_samples.append({
                'original_policy': policy_excerpt,
                'perturbed_policy': perturbed_policy,
                'question': question,
                'original_answer': answer,
                'predicted_answer': predicted_answer,
                'perturbation_method': method,
                'perturbed_word': word,
                'word_position': word_idx
            })
    
    return perturbed_samples

input_file = 'qa_dataset.csv' 
output_file = 'qa_perturbed_results.csv'

while True:
    method = input("Choose perturbation method (replace/swap/delete/insert/repeat/keyboard): ").lower()
    if method in ['replace', 'swap', 'delete', 'insert', 'repeat', 'keyboard']:
        break
    print("Invalid input. Please enter a valid method.")

df = pd.read_csv(input_file)
all_results = []

for idx, row in df.iterrows():
    print(f"Processing sample {idx + 1}/{len(df)}")
    
    policy_excerpt = row['policy_excerpt']
    question = row['question']
    answer = row['answer']
    
    perturbed_samples = process_qa_sample(policy_excerpt, question, answer, method)
    all_results.extend(perturbed_samples)

results_df = pd.DataFrame(all_results)
results_df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
print(f"Total perturbed samples generated: {len(all_results)}")