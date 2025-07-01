import pandas as pd
import numpy as np
import torch
# from transformers import DistilBertModel, DistilBertTokenizer
from transformers import BertTokenizer, BertModel



device = 'cuda' if torch.cuda.is_available() else 'cpu'

# dowload online (run locally on laptop on filesystem mounted with sshfs)
model = BertModel.from_pretrained('google/bert_uncased_L-12_H-768_A-12')
tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')

print(model)

# model.save_pretrained("models/bert_base")
# tokenizer.save_pretrained("models/bert_tiny_tokenizer")

# assert False

# load on supercloud
# model = DistilBertModel.from_pretrained("models/distilbert")
# tokenizer = DistilBertTokenizer.from_pretrained("models/distilbert_tokenizer")

# load dataset
dataset = pd.read_csv("twitter_sentiment.csv", encoding='latin-1', names=['target', 'id', 'date', 'query', 'username', 'text'])
dataset = dataset.iloc[:10, :] # TODO: For now for debugging

# tokenize tweets and pad to equal length, attention mask: tell what is padding
tokenized_reviews = dataset.text.apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
max_len = max(map(len,tokenized_reviews))
padded_reviews = np.array([ i+[0]*(max_len-len(i))  for i in tokenized_reviews])
attention_masked_reviews = np.where(padded_reviews!=0,1,0)

# get last hidden states as embedding
input_ids = torch.tensor(padded_reviews).to(device)
attention_mask = torch.tensor(attention_masked_reviews).to(device)
model.to(device)
with torch.no_grad():
  last_hidden_states = model(input_ids, attention_mask=attention_mask)


# Note: For classification, it's okay to only use the first embedded token ("CLS")
X = last_hidden_states[0][:,0,:] #.cpu().numpy()
y = torch.tensor(dataset.target)

print(type(X), X.shape)
print(type(y), y.shape)
