import torch
import torch.nn as nn

#learned token embedding, positional embedding (removed segment embedding)


class Embedding(nn.Module):
	def __init__(self, vocab_size, d_model, maxlen, n_segments=2):
		super(Embedding, self).__init__()
		self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
		self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
		#self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
		#self.norm = nn.LayerNorm(d_model)

	def forward(self, x, seg = None):
		seq_len = x.size(1)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		pos = torch.arange(seq_len, dtype=torch.long)
		pos = pos.unsqueeze(0).expand_as(x).to(device)  # (seq_len,) -> (batch_size, seq_len)

		embedding = self.tok_embed(x) + self.pos_embed(pos)# + self.seg_embed(seg)
		#return self.norm(embedding)
		return embedding
   



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
		self.W_Q = nn.Linear(d_model, self.d_head * self.n_heads, bias=False)
		self.W_K = nn.Linear(d_model, self.d_head * self.n_heads, bias=False)
		self.W_V = nn.Linear(d_model, self.d_head * self.n_heads, bias=False)

		self.norm = nn.LayerNorm(d_model)

	def forward(self, enc_input, attn_mask):
	   	# q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
		residual = enc_input
		batch_size = enc_input.size(0)
		sentence_len = enc_input.size(1)
	   	# (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
		q_s = self.W_Q(enc_input).reshape(batch_size, sentence_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)	# q_s: [batch_size x n_heads x len_q x d_k]
		k_s = self.W_K(enc_input).reshape(batch_size, sentence_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)  # k_s: [batch_size x n_heads x len_k x d_k]
		v_s = self.W_V(enc_input).reshape(batch_size, sentence_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)  # v_s: [batch_size x n_heads x len_k x d_v]

		attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

		# context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
		context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask, self.d_head)
		context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_head) # context: [batch_size x len_q x n_heads * d_v]
		#output = nn.Linear(self.n_heads * self.d_head, d_model)(context)
		# output: [batch_size x len_q x d_model]

		return self.norm(context + residual), attn


class ScaledDotProductAttention(nn.Module):		
	def __init__(self):
		super(ScaledDotProductAttention, self).__init__()

	def forward(self, Q, K, V, attn_mask, d_head):
		scores = torch.matmul(Q, K.transpose(-1, -2)) / (d_head**0.5) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
		scores.masked_fill_(attn_mask, float('-inf')) # Fills elements of self tensor with value where mask is one.
		attn = nn.Softmax(dim=-1)(scores)
		context = torch.matmul(attn, V)
		return context, attn
   

class BERT(nn.Module):
	def __init__(self, vocab_size, d_model, n_layers, maxlen, ff_exp, heads, n_labels):
		super(BERT, self).__init__()
		self.embedding = Embedding(vocab_size, d_model, maxlen)
		
		self.layers = nn.ModuleList([EncoderLayer(d_model, ff_exp, heads) for _ in range(n_layers)])
		

		self.classifier = nn.Sequential(
									nn.Linear(d_model, d_model),
									nn.Tanh(),
									nn.Linear(d_model, n_labels)
							)
		


		self.masker = nn.Sequential(
			nn.Linear(d_model, d_model),
			nn.ReLU(),
			nn.LayerNorm(d_model)
		)
		
		# decoder is shared with embedding layer
		embed_weight = self.embedding.tok_embed.weight
		n_vocab, n_dim = embed_weight.size()
		self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
		self.decoder.weight = embed_weight
		self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

	def forward(self, input_ids, masked_pos, segment_ids = None):
		output = self.embedding(input_ids)
		enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
		for layer in self.layers:
			output, enc_self_attn = layer(output, enc_self_attn_mask)
		# output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
		
		# classification will be decided by first token(CLS)
		logits_clsf = self.classifier(output[:, 0]) # [batch_size, d_model] -> [batch_size, 2]

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)).to(torch.int64).to(device) # [batch_size, max_pred, d_model]

		# get masked position from final output of transformer.
		h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
		h_masked = self.masker(h_masked)
		logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]

		return logits_lm, logits_clsf
	

def get_attn_pad_mask(seq_q, seq_k):
	batch_size, len_q = seq_q.size()
	batch_size, len_k = seq_k.size()
	# eq(zero) is PAD token
	pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
	return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k