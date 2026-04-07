import pandas as pd
import random
import string
import spacy

# Load the spaCy model for POS tagging and dependency parsing
nlp = spacy.load('en_core_web_sm')

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
        perturbations.append(new_word)
    return perturbations

def find_action_verb_or_noun(doc):
    action_verbs = [token for token in doc if token.pos_ == 'VERB' and token.dep_ in ('ROOT', 'acl', 'advcl') and token.text != '<mask>']
    if action_verbs:
        return action_verbs[0].text, action_verbs[0].idx
    nouns = [token for token in doc if token.pos_ == 'NOUN' and token.text != '<mask>']
    if nouns:
        return nouns[0].text, nouns[0].idx
    return None, None

def process_context(context, method):
    parts = context.split('<mask>')
    if len(parts) != 2:
        return [context]

    doc = nlp(context)
    target_word, target_idx = find_action_verb_or_noun(doc)

    if not target_word:
        return [context]

    perturbations = perturb_word(target_word, method)
    new_contexts = []

    for perturbed_word in perturbations:
        new_context = context[:target_idx] + perturbed_word + context[target_idx + len(target_word):]
        new_contexts.append(new_context)

    return new_contexts if new_contexts else [context]

input_file = ''
output_file = ''

while True:
    method = input("Choose perturbation method (replace/swap/delete/insert/repeat/keyboard): ").lower()
    if method in ['replace', 'swap', 'delete', 'insert', 'repeat', 'keyboard']:
        break
    print("Invalid input. Please enter 'replace', 'swap', 'delete', 'insert', 'repeat', 'keyboard'.")

#input csv
df = pd.read_csv(input_file)
new_rows = []

for _, row in df.iterrows():
    perturbed_contexts = process_context(row['context'], method)
    for perturbed_context in perturbed_contexts:
        new_row = row.copy()
        new_row['adversarial_text'] = perturbed_context
        new_rows.append(new_row)

new_df = pd.DataFrame(new_rows)
new_df.to_csv(output_file, index=False)
