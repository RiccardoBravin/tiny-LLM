import torch
import torch.nn as nn
import math

#learned token embedding, positional embedding (removed segment embedding)
#multihead attention with scaled dot product attention
#feed forward with relu activation
#layer normalization


def n_ary_gray_code(n, base = 3):
    # n x n**3 list 
    gray = [[0] * n for _ in range(base**n)]
    for j in range(n):
        i = 0
        val = 0
        invert = True
        while i < base**n:
            for k in range(base**j):
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
	def __init__(self, vocab_size, d_model, red_d_model, maxlen, dropout=0.1):
		super(Embedding, self).__init__()

		self.tok_embed = nn.Embedding(vocab_size, red_d_model)
		
		log_len = math.ceil(math.log(maxlen) / math.log(3))
		self.expand_layer = nn.Linear(red_d_model + log_len, d_model)

		base_3_representation = n_ary_gray_code(log_len, base=3)[:maxlen]
		self.pos_embed = torch.tensor(base_3_representation) - 1 
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		batch_size, seq_length = x.shape
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		word_embeddings = self.tok_embed(x)
		pos_embeddings = self.pos_embed.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
		embedding_red = torch.cat((word_embeddings, pos_embeddings), dim = 2)
		embedding = self.expand_layer(embedding_red)
		return self.dropout(embedding) 


   

class dumbclassifier(nn.Module):
	def __init__(self, vocab_size, d_model, red_d_model, n_layers, maxlen, ff_exp, heads, n_labels):
		super(dumbclassifier, self).__init__()
		self.embedding = Embedding(vocab_size, d_model, red_d_model, maxlen)
	
		self.shuffler = nn.Sequential(
									nn.Linear(maxlen, maxlen),
									nn.Tanh(),
									nn.Linear(maxlen, 1)
							)
		
		self.classifier = nn.Sequential(
									nn.Linear(d_model, d_model),
									nn.Tanh(),
									nn.Linear(d_model, n_labels)
							)
		



	def forward(self, input_ids, masked_pos, segment_ids = None):
		output = self.embedding(input_ids)
		
		# output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
		mid = self.shuffler(output.transpose(1, 2)).squeeze(2)
        
		# classification will be decided by first token(CLS)
		logits_clsf = self.classifier(mid) # [batch_size, d_model] -> [batch_size, 2]


		return 0, logits_clsf
	

