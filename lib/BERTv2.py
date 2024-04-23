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


class EncoderLayer(nn.Module):
	def __init__(self, embed_dim=128, forward_expansion=4, n_heads=8, dropout=0.1):
		super(EncoderLayer, self).__init__()
		self.enc_self_attn = MultiHeadAttention(embed_dim, n_heads)
		self.ffn = nn.Sequential(
        	nn.Linear(embed_dim, int(forward_expansion * embed_dim)),
        	nn.ReLU(), #using relu because it is more efficient than gelu
        	nn.Linear(int(forward_expansion * embed_dim), embed_dim)
    	)

	def forward(self, enc_inputs, enc_self_attn_mask):
		enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
		enc_outputs = self.ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
		return enc_outputs, attn

class MultiHeadAttention(nn.Module):
	def __init__(self, d_model, n_heads):
		super(MultiHeadAttention, self).__init__()
		self.n_heads = n_heads
		
		assert(d_model % n_heads == 0)
		self.d_head = d_model // n_heads
		self.W_Q = nn.Linear(d_model, d_model, bias=False)
		self.W_O = nn.Linear(d_model, d_model, bias=False)

		self.norm = nn.LayerNorm(d_model)

	def forward(self, enc_input, attn_mask):
	   	# q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
		batch_size = enc_input.size(0)
		sentence_len = enc_input.size(1)
		d_model = enc_input.size(2)
	   	# (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
		q_s = self.W_Q(enc_input)# q_s: [batch_size x s_len x d_model]

		attention_scores = torch.matmul(q_s, torch.transpose(enc_input,1,2)) / (d_model ** (0.5))
		attention_scores.masked_fill_(attn_mask, float('-inf'))

		attention_dist = torch.softmax(attention_scores, dim=-1)

		attention_out = torch.matmul(attention_dist, enc_input)

		out = self.W_O(attention_out)

		#attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

		# context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
		# context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask, self.d_head)
		# context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_head) # context: [batch_size x len_q x n_heads * d_v]
		# #output = nn.Linear(self.n_heads * self.d_head, d_model)(context)
		# # output: [batch_size x len_q x d_model]
		# out = self.W_O(attention) # out: [batch_size x s_len x d_model]

		return self.norm(out + enc_input), None


class ScaledDotProductAttention(nn.Module):		
	def __init__(self):
		super(ScaledDotProductAttention, self).__init__()

	def forward(self, Q, K, V, attn_mask, d_head):
		scores = torch.matmul(Q, K.transpose(-1, -2)) / (d_head**0.5) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
		scores.masked_fill_(attn_mask, float('-inf')) # Fills elements of self tensor with value where mask is one.
		attn = nn.Softmax(dim=-1)(scores)
		context = torch.matmul(attn, V)
		return context, attn
   

class BERTv2(nn.Module):
	def __init__(self, vocab_size, d_model, red_d_model, n_layers, maxlen, ff_exp, heads, n_labels):
		super(BERTv2, self).__init__()
		self.embedding = Embedding(vocab_size, d_model, red_d_model, maxlen)
		
		self.layers = nn.ModuleList([EncoderLayer(d_model, ff_exp, heads) for _ in range(n_layers)])
		

		self.classifier = nn.Sequential(
									nn.Linear(d_model, d_model),
									nn.Tanh(),
									nn.Linear(d_model, n_labels)
							)
		


		# self.masker = nn.Sequential(
		# 	nn.Linear(d_model, d_model),
		# 	nn.ReLU(),
		# 	nn.LayerNorm(d_model)
		# )
		
		# removed decoder layer because it is not needed for classification

		# decoder is shared with embedding layer
		# embed_weight = self.embedding.tok_embed.weight
		
		# self.decoder = nn.Linear(d_model, vocab_size, bias=False)
		# self.decoder.weight = embed_weight
		# self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))

	def forward(self, input_ids, masked_pos, segment_ids = None):
		output = self.embedding(input_ids)
		enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
		for layer in self.layers:
			output, enc_self_attn = layer(output, enc_self_attn_mask)
		# output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
		
		# classification will be decided by first token(CLS)
		logits_clsf = self.classifier(output[:, 0]) # [batch_size, d_model] -> [batch_size, 2]

		# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		# masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)).to(torch.int64).to(device) # [batch_size, max_pred, d_model]

		# get masked position from final output of transformer.
		# h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
		# h_masked = self.masker(h_masked)
		#logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]

		return 0, logits_clsf
	

def get_attn_pad_mask(seq_q, seq_k):
	batch_size, len_q = seq_q.size()
	batch_size, len_k = seq_k.size()
	# eq(zero) is PAD token
	pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
	return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k