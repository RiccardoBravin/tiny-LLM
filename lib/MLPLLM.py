import torch
import torch.nn as nn

import math

from lib.utils import pad_sequences

# embedder con posizionale in base 3 e upscaling del token embedding conncatenato al posizionale
# efficient attention con swiglu
# fully connected con max pooling (circa)
# redone positional embedding to make nearer positions nearer in the embedding space


def n_ary_gray_code(n, base = 3):
    # n x n**3 list 
    gray = [[0] * n for _ in range(base**n)]
    for j in range(n):
        i = 0
        val = 0
        invert = True
        while i < base**n:
            for k in range(base**j):
                # print(i+k)
                gray[i+k][j] = val
            
            i += base**j
            
            
            if  invert:
                val += 1
            else:
                val -= 1
            
            if val == base:
                invert = not invert
                val = base - 1
            elif val == -1:
                invert = not invert
                val = 0


    return gray

class Embedding(nn.Module):
	def __init__(self, vocab_size, max_length, embed_dim, reduced_embed = 16, dropout=0.1):
		super(Embedding, self).__init__()

		log_len = math.ceil(math.log(max_length) / math.log(3))
		self.word_embed = nn.Embedding(vocab_size, reduced_embed)
		self.expand_layer = nn.Linear(reduced_embed + log_len, embed_dim)

		base_3_representation = n_ary_gray_code(log_len, base=3)[:max_length]
		self.pos_embed = torch.tensor(base_3_representation) - 1 
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		batch_size, seq_length = x.shape
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		word_embeddings = self.word_embed(x)
		pos_embeddings = self.pos_embed.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
		embedding_red = torch.cat((word_embeddings, pos_embeddings), dim = 2)
		embedding = self.expand_layer(embedding_red)
		return self.dropout(embedding) 


	

class EfficientAttention(nn.Module):
	def __init__(self, embed_dim, num_heads):
		super(EfficientAttention, self).__init__()
		self.embed_dim = embed_dim
		# self.num_heads = num_heads
		# self.head_dim = embed_dim // num_heads

		# assert (self.num_heads*self.head_dim == self.embed_dim),'embed size must be divisible by number of heads'

		self.w_queries = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
		self.w_output = nn.Linear(self.embed_dim, self.embed_dim, bias=False)


	def forward(self, x):

		# shape of x = [batch_size, sentence_length, embedding_dim]
		batch_size = x.shape[0]
		sentence_len = x.shape[1]

		queries = self.w_queries(x) #shape [batch_size, sentence_length, embedding_dim]

		attention_scores = torch.einsum('bij,bjk->bik', queries, torch.transpose(x,1,2))

		attention_dist = torch.softmax(attention_scores / (self.embed_dim ** (0.5)), dim=-1)

		attention_out = torch.einsum('bij,bjk->bik', attention_dist, x)

		out = self.w_output(attention_out)

		return out
	

class TransformerEncoder(nn.Module):
	def __init__(self, embed_dim, num_heads, forward_expansion, dropout=0.1):
		super(TransformerEncoder, self).__init__()

		self.attention = EfficientAttention(embed_dim, num_heads)
		self.norm1 = nn.LayerNorm(embed_dim)
		self.norm2 = nn.LayerNorm(embed_dim)

		self.fc_up1 = nn.Linear(embed_dim, int(forward_expansion*embed_dim))
		self.fc_up2 = nn.Linear(embed_dim, int(forward_expansion*embed_dim))
		self.fc_down = nn.Linear(int(forward_expansion*embed_dim), embed_dim)
		self.silu = torch.nn.SiLU(inplace=False)

		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		attention_out = self.dropout(self.attention(x))
		x = self.norm1(x + attention_out)
		mid1 = self.fc_up1(x)
		mid2 = self.silu(self.fc_up2(x))
		mid = torch.mul(mid1,mid2)
		swiglu = self.fc_down(mid)
		forward_out = self.dropout(swiglu)
		out = self.norm2(x + forward_out)

		return out
	

class MLPLLM(nn.Module):
	def __init__(self, vocab_size, max_length, red_embed_dim, embed_dim, num_heads, forward_expansion, layers, out_labels):
			super(MLPLLM, self).__init__()

			self.embedder = Embedding(vocab_size, max_length, embed_dim, red_embed_dim)
			#self.encoder = nn.Sequential(*[TransformerEncoder(embed_dim, num_heads, forward_expansion) for _ in range(layers)])
			self.l1 = nn.Linear(embed_dim, embed_dim)
			self.l2 = nn.Linear(max_length, max_length)
			self.fc = nn.Linear(embed_dim, out_labels)
			self.af = nn.ReLU()
	def forward(self, x):
		embedding = self.embedder(x)
		x = self.l1(embedding)
		x = self.af(x)
		x = self.l2(x.transpose(1,2))
		x = self.af(x)
		x = x.transpose(1,2)
		
		compact_encoding = x.max(dim=1)[0]
		out = self.fc(compact_encoding)
		return torch.sigmoid(out)
	

