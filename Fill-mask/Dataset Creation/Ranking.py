#Dataset using Ranking (better using gpu)
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
import tensorflow_hub as hub
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
import spacy


nltk.download('stopwords')


nlp = spacy.load("en_core_web_sm")

nltk_stopwords = set(stopwords.words('english'))
spacy_stopwords = nlp.Defaults.stop_words
combined_stopwords = nltk_stopwords.union(spacy_stopwords)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NLI_infer_BERT(nn.Module):
    def __init__(self, pretrained_dir, nclasses=2, max_seq_length=128):
        super(NLI_infer_BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).to(get_device())
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir)
        self.max_seq_length = max_seq_length

    def text_pred(self, text_data, batch_size=32):
        self.model.eval()
        inputs, masks = self.encode_texts(text_data)
        dataset = TensorDataset(inputs, masks)
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)
        probs_all = []
        for batch in dataloader:
            batch = tuple(t.to(get_device()) for t in batch)  # Ensure batch is on the correct device (GPU/CPU)
            inputs, masks = batch
            with torch.no_grad():
                outputs = self.model(inputs, attention_mask=masks)
                logits = outputs.logits
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)
        return torch.cat(probs_all, dim=0)

    def encode_texts(self, texts):
        inputs = []
        masks = []
        for text in texts:
            encoded_dict = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding='max_length',  
                return_attention_mask=True,
                return_tensors='pt',
            )
            inputs.append(encoded_dict['input_ids'])
            masks.append(encoded_dict['attention_mask'])
        inputs = torch.cat(inputs, dim=0)
        masks = torch.cat(masks, dim=0)
        return inputs, masks

def find_most_important_word(text_ls, original_text, predictor, stop_words_set, batch_size=32):
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()

    len_text = len(text_ls)
    leave_1_texts = [text_ls[:ii] + ['<mask>'] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
    leave_1_probs = predictor(leave_1_texts, batch_size=batch_size)

    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
    import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
                leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))).data.cpu().numpy()

    # Rank words by importance and filter out stop words
    ranked_words = [
    (idx, word, score)
    for idx, (word, score) in enumerate(zip(text_ls, import_scores))
    if word.lower() not in combined_stopwords and re.match(r'^[a-zA-Z0-9]+$', word)
    ]
    ranked_words.sort(key=lambda x: x[2], reverse=True)

 
    most_important_idx = ranked_words[0][0]
    most_important_word = ranked_words[0][1]


    start_pos = [i for i in range(len(original_text)) if original_text.startswith(most_important_word, i)]

    return most_important_idx, start_pos, most_important_word

def mask_and_save(csv_path, context_column, target_model_path, output_file, batch_size=32, max_seq_length=128):
    df = pd.read_csv(csv_path)
    contexts = df[context_column].tolist()
    data = [context.split() for context in contexts]  # Split context into tokens

    print("Building BERT Model...")
    model = NLI_infer_BERT(target_model_path, max_seq_length=max_seq_length)
    predictor = model.text_pred
    print("BERT Model built!")

    # Start masking important words
    masked_texts = []
    answers = []
    stop_words_set = combined_stopwords
    print('Start masking important words!')

    for idx, (text, original_text) in enumerate(zip(data, contexts)):
        if idx % 20 == 0:
            print(f'{idx} samples out of {len(data)} have been finished!')
        most_important_idx, start_pos, most_important_word = find_most_important_word(text, original_text, predictor, stop_words_set, batch_size=batch_size)

        # Apply <mask> to the most important word
        text[most_important_idx] = '<mask>'
        masked_text = ' '.join(text)
        masked_texts.append(masked_text)

        answers.append({'answer_start': start_pos, 'text': [most_important_word]})

    df['masked_context'] = masked_texts
    df['answer'] = answers

   
    df.to_csv(output_file, index=False)
    print(f"Masking complete. Results saved to {output_file}.")


mask_and_save(
    csv_path="", #input path
    context_column="purified_context",
    target_model_path="bert-base-uncased",
    output_file="", #output path
    batch_size=32,
    max_seq_length=128
)

