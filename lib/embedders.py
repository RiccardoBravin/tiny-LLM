import torch
from torch import nn


class STD_Embedder(nn.Module):
	def __init__(self, vocab_size, max_length, hidden_size, segments = 3):
		r"""
		Standard Embedder for transformer models taken from BERT. It is a combination of a token and a positional embedding
		
		Args:
			vocab_size: the size of the vocabulary
			embed_size: the size of the embeddings in output
			max_len: the maximum length of the sequence
			dropout: the dropout rate
		"""

		super().__init__()
		# (m, seq_len) --> (m, seq_len, embed_size)
		# padding_idx is not updated during training, remains as fixed pad (0)
		self.token = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=0)
		self.position = torch.nn.Embedding(max_length, hidden_size)
		self.sentence = torch.nn.Embedding(3, hidden_size, padding_idx=0)


	def forward(self, tokens):
		#build the positions tensor of the tokens
		positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0).expand_as(tokens)


		sep_pos = torch.nonzero(tokens == 4)        
		sentence_ids = torch.ones_like(tokens, device=tokens.device)
		for i in sep_pos:
			sentence_ids[i[0], i[1]:] = 2
		
		sentence_ids[tokens == 0] = 0


		x = self.token(tokens) + self.position(positions) + self.sentence(sentence_ids)
		return x
	


class Nano_Embedder(nn.Module):
	def __init__(self, vocab_size, max_len, hidden_size, reduced_embedding, segments = 3):
		r"""
		Embedder for transformer models taken from NanoBERT. It is a combination of a token and a positional embedding where the token embedding
		is done in a lower dimension and then projected to the desired dimension with a linear layer
		
		Args:
			vocab_size: the size of the vocabulary
			embed_size: the size of the embeddings in output
			red_embed_size: the size of the embeddings in the lower dimension
			max_len: the maximum length of the sequence
		"""

		super().__init__()
		# (m, seq_len) --> (m, seq_len, embed_size)
		# padding_idx is not updated during training, remains as fixed pad (0)
		self.token = torch.nn.Embedding(vocab_size,reduced_embedding, padding_idx=0)
		self.tok_expander = torch.nn.Linear(reduced_embedding, hidden_size, bias=False)
		
		self.position = torch.nn.Embedding(max_len,reduced_embedding)
		self.pos_expander = torch.nn.Linear(reduced_embedding, hidden_size, bias=False)

		self.sentence = torch.nn.Embedding(segments, hidden_size, padding_idx=0)

	   
	def forward(self, tokens):
		#build the positions tensor of the tokens
		positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0).expand_as(tokens)

		sep_pos = torch.nonzero(tokens == 4)        
		sentence_ids = torch.ones_like(tokens, device=tokens.device)

		for i in sep_pos:
			sentence_ids[i[0], i[1]:] = 2
		
		sentence_ids[tokens == 0] = 0


		x = self.tok_expander(self.token(tokens)) + self.pos_expander(self.position(positions)) + self.sentence(sentence_ids)
		return x
	