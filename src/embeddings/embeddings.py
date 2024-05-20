import torch
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import normalize

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained('bert-base-uncased')

def tokenize(text):
    encoded_inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True, max_length=512)
    return encoded_inputs

def create_embeddings(tokenized_data):
    input_ids = tokenized_data['input_ids']
    attention_mask = tokenized_data['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings.numpy()
