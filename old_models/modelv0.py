import torch
import torch.nn as nn

import math

from lib.utils import pad_sequences


class Embedding(nn.Module):
  def __init__(self, vocab_size, max_length, embed_dim, dropout=0.1):
    super(Embedding, self).__init__()
    self.word_embed = nn.Embedding(vocab_size, embed_dim)
    self.pos_embed = nn.Embedding(max_length, embed_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    batch_size, seq_length = x.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    positions = torch.arange(0, seq_length).expand(batch_size, seq_length).to(device)
    embedding = self.word_embed(x) + self.pos_embed(positions)
    #print(embedding.shape)
    return self.dropout(embedding)
  

class MHSelfAttention(nn.Module):
  def __init__(self, embed_dim, num_heads):
    super(MHSelfAttention, self).__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads

    assert (self.num_heads*self.head_dim == self.embed_dim), \
    'embed size must be divisible by number of heads'

    self.w_queries = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
    self.w_keys = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
    self.w_values = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    self.fc_out = nn.Linear(self.head_dim*self.num_heads , self.embed_dim)

  def forward(self, x):

    # shape of x = [batch_size, sentence_length, embedding_dim]
    batch_size = x.shape[0]
    sentence_len = x.shape[1]

    queries = self.w_queries(x).reshape(
        batch_size, sentence_len, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3)

    keys = self.w_keys(x).reshape(
        batch_size, sentence_len, self.num_heads, self.head_dim).permute(
            0, 2, 3, 1)


    values = self.w_values(x).reshape(
        batch_size, sentence_len, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3)

    attention_scores = torch.einsum('bijk,bikl->bijl', queries, keys)
    attention_dist = torch.softmax(attention_scores /
                               (self.embed_dim ** (1/2)), dim=-1)
    attention_out = torch.einsum('bijk,bikl->bijl', attention_dist, values)
    concatenated_out = attention_out.permute(0, 2, 1, 3).reshape(
        batch_size, sentence_len, self.embed_dim)

    return concatenated_out
  

class TransformerEncoder(nn.Module):
  def __init__(self, embed_dim, num_heads, forward_expansion, dropout=0.1):
    super(TransformerEncoder, self).__init__()

    self.attention = MHSelfAttention(embed_dim, num_heads)
    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)

    self.feed_forward = nn.Sequential(
        nn.Linear(embed_dim, forward_expansion*embed_dim),
        nn.ReLU(),
        nn.Linear(forward_expansion*embed_dim, embed_dim)
    )

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    attention_out = self.dropout(self.attention(x))
    x = self.norm1(x + attention_out)
    forward_out = self.dropout(self.feed_forward(x))
    out = self.norm2(x + forward_out)

    return out
  
class Classifier(nn.Module):
  def __init__(self, vocab_size, max_length, embed_dim, num_heads, forward_expansion, out_labels):
      super(Classifier, self).__init__()

      self.embedder = Embedding(vocab_size, max_length, embed_dim)
      self.encoder = TransformerEncoder(embed_dim, num_heads, forward_expansion)
      self.fc = nn.Linear(embed_dim, out_labels)

  def forward(self, x):
    embedding = self.embedder(x)
    encoding = self.encoder(embedding)
    compact_encoding = encoding.max(dim=1)[0]
    out = self.fc(compact_encoding)
    return torch.sigmoid(out)