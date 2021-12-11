import gzip
import json
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel

bert_embedding_layer = BertModel.from_pretrained('bert-base-uncased')
print('bert_embedding_layer: ', bert_embedding_layer)
    

embedding_file = "data/datasets/R2R_VLNCE_v1-2_preprocessed/embeddings.json.gz"
with gzip.open(embedding_file, "rt") as f:
    pre_embeddings = torch.tensor(json.load(f))            
glove_embedding_layer = nn.Embedding.from_pretrained(
    embeddings=pre_embeddings,
    freeze=True,
)
print('glove_embedding_layer: ', glove_embedding_layer)

