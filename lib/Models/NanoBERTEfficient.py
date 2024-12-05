import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from lib.embedders import Nano_Embedder
from lib.attentions import EfficientAttention
from lib.utils import make_score_mask

class NanoBERTEfficient_Config(PretrainedConfig):
	model_type = "NanoBERTEfficient"
	def __init__(self, reduced_embedding:int=16,forward_expansion:float=1, max_length:int=256, **kwargs):
		super().__init__(**kwargs)
		self.reduced_embedding = reduced_embedding
		self.forward_expansion = forward_expansion
		self.max_length = max_length
		



class NanoBERTEfficient(PreTrainedModel):
	config_class = NanoBERTEfficient_Config
	def __init__(self, config:NanoBERTEfficient_Config):
		super().__init__(config)
		self.embedder = Nano_Embedder(
								vocab_size=config.vocab_size, 
								max_len=config.max_length, 
								hidden_size=config.hidden_size, 
								reduced_embedding=config.reduced_embedding,
								segments=3
						)
		
		self.pre_norms = nn.ModuleList([
							nn.RMSNorm(config.hidden_size) for _ in range(config.num_hidden_layers)
						])
		
		self.attentions = nn.ModuleList([
							EfficientAttention(
									hidden_size=config.hidden_size,
							) for _ in range(config.num_hidden_layers)
						])	
		
		self.post_norms = nn.ModuleList([
							nn.RMSNorm(config.hidden_size) for _ in range(config.num_hidden_layers)
						])
		
		self.feed_forward = nn.ModuleList([
							nn.Sequential(
								nn.Linear(config.hidden_size, int(config.hidden_size*config.forward_expansion)),
								nn.SiLU(),
								nn.Linear(int(config.hidden_size*config.forward_expansion), config.hidden_size),
							) for _ in range(config.num_hidden_layers)
						])

	
	def forward(self, input, mask):
		mask = make_score_mask(mask)

		x = self.embedder(input)
		for pre_norm, attention, post_norm, feed_forward in zip(self.pre_norms, self.attentions, self.post_norms, self.feed_forward):
			x = pre_norm(x)	
			x = attention(x, mask)
			x = post_norm(x)
			x = feed_forward(x) + x
		return x