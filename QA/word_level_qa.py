import pandas as pd
import random
import spacy
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nlp = spacy.load('en_core_web_sm')

model_name = ""
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
model.eval()

def get_synonyms(word, pos=None):
    
    synonyms = set()
    pos_map = {
        'NOUN': wordnet.NOUN,
        'VERB': wordnet.VERB,
        'ADJ': wordnet.ADJ,
        'ADV': wordnet.ADV
    }
    
    wordnet_pos = pos_map.get(pos, None) if pos else None
    if wordnet_pos:
        synsets = wordnet.synsets(word, pos=wordnet_pos)
    else:
        synsets = wordnet.synsets(word)
    
    for syn in synsets:
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower():
                synonyms.add(synonym)
    
    return list(synonyms)

def word_repetition_attack(word, position='before'):
    if position == 'before':
        return f"{word} {word}"
    else: 
        return f"{word} {word}"

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
    start_score = torch.max(torch.softmax(outputs.start_logits, dim=1)).item()
    end_score = torch.max(torch.softmax(outputs.end_logits, dim=1)).item()
    confidence = (start_score + end_score) / 2
    
    return predicted_answer, confidence

def synonym_replacement_attack(policy_excerpt, question, answer, start_idx, end_idx):
    answer_text = policy_excerpt[start_idx:end_idx]
    doc = nlp(answer_text)
    
    perturbed_samples = []
    
    for token_idx, token in enumerate(doc):
        word = token.text
        pos = token.pos_
        if pos == 'PUNCT' or len(word) <= 2:
            continue
        synonyms = get_synonyms(word, pos)
        
        if not synonyms:
            continue
        synonyms = synonyms[:5]
        
        for synonym in synonyms:
            perturbed_words = [t.text for t in doc]
            perturbed_words[token_idx] = synonym
            perturbed_answer_text = ' '.join(perturbed_words)
            perturbed_policy = (
                policy_excerpt[:start_idx] + 
                perturbed_answer_text + 
                policy_excerpt[end_idx:]
            )

            predicted_answer, confidence = get_qa_prediction(question, perturbed_policy)
            
            perturbed_samples.append({
                'original_policy': policy_excerpt,
                'perturbed_policy': perturbed_policy,
                'question': question,
                'original_answer': answer,
                'perturbed_answer_text': perturbed_answer_text,
                'predicted_answer': predicted_answer,
                'confidence': confidence,
                'perturbation_method': 'synonym_replacement',
                'original_word': word,
                'perturbed_word': synonym,
                'word_position': token_idx,
                'word_pos': pos
            })
    
    return perturbed_samples

def word_repetition_attack_qa(policy_excerpt, question, answer, start_idx, end_idx):
    answer_text = policy_excerpt[start_idx:end_idx]
    doc = nlp(answer_text)
    
    perturbed_samples = []
    
    for token_idx, token in enumerate(doc):
        word = token.text
        pos = token.pos_
        if pos == 'PUNCT':
            continue
        repeated_word = word_repetition_attack(word, position='before')
        perturbed_words = [t.text for t in doc]
        perturbed_words[token_idx] = repeated_word
        perturbed_answer_text = ' '.join(perturbed_words)
        perturbed_policy = (
            policy_excerpt[:start_idx] + 
            perturbed_answer_text + 
            policy_excerpt[end_idx:]
        )
        predicted_answer, confidence = get_qa_prediction(question, perturbed_policy)
        
        perturbed_samples.append({
            'original_policy': policy_excerpt,
            'perturbed_policy': perturbed_policy,
            'question': question,
            'original_answer': answer,
            'perturbed_answer_text': perturbed_answer_text,
            'predicted_answer': predicted_answer,
            'confidence': confidence,
            'perturbation_method': 'word_repetition',
            'original_word': word,
            'perturbed_word': repeated_word,
            'word_position': token_idx,
            'word_pos': pos
        })
    
    return perturbed_samples

def process_qa_sample(policy_excerpt, question, answer, method):
    start_idx, end_idx = find_answer_in_context(policy_excerpt, answer)
    
    if start_idx is None:
        print(f"Warning: Answer '{answer}' not found in policy excerpt")
        return []
    
    if method == 'synonym':
        return synonym_replacement_attack(policy_excerpt, question, answer, start_idx, end_idx)
    elif method == 'repetition':
        return word_repetition_attack_qa(policy_excerpt, question, answer, start_idx, end_idx)
    elif method == 'both':
        synonym_samples = synonym_replacement_attack(policy_excerpt, question, answer, start_idx, end_idx)
        repetition_samples = word_repetition_attack_qa(policy_excerpt, question, answer, start_idx, end_idx)
        return synonym_samples + repetition_samples
    else:
        return []

input_file = 'qa_dataset.csv' 
output_file = 'qa_word_level_attacks.csv'

while True:
    method = input("Choose word-level attack (synonym/repetition): ").lower()
    if method in ['synonym', 'repetition']:
        break
    print("Invalid input. Please enter 'synonym', 'repetition'.")


df = pd.read_csv(input_file)
all_results = []

for idx, row in df.iterrows():
    print(f"Processing sample {idx + 1}/{len(df)}")
    
    policy_excerpt = row['policy_excerpt']
    question = row['question']
    answer = row['answer']
    
    perturbed_samples = process_qa_sample(policy_excerpt, question, answer, method)
    all_results.extend(perturbed_samples)
    
    if (idx + 1) % 10 == 0:
        print(f"Generated {len(all_results)} perturbed samples so far...")


results_df = pd.DataFrame(all_results)
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to {output_file}")
print(f"Total perturbed samples generated: {len(all_results)}")