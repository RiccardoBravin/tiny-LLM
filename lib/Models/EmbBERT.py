import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from lib.embedders import Nano_Embedder
from lib.attentions import EfficientDifferentialSkipAttention
from lib.utils import make_score_mask

class EmbBERT_Config(PretrainedConfig):
	model_type = "EmbBERT"
	def __init__(self, reduced_embedding:int=16,forward_expansion:float=1, kernel_size:int=32, max_length:int=256, variation:str="", **kwargs):
		super().__init__(**kwargs)
		self.reduced_embedding = reduced_embedding
		self.forward_expansion = forward_expansion
		self.kernel_size = kernel_size
		self.max_length = max_length
		self.variation = variation
		



class EmbBERT(PreTrainedModel):
	config_class = EmbBERT_Config
	def __init__(self, config:EmbBERT_Config):
		super().__init__(config)
		self.embedder = Nano_Embedder(
								vocab_size=config.vocab_size, 
								max_len=config.max_length, 
								hidden_size=config.hidden_size, 
								reduced_embedding=config.reduced_embedding,
								segments=3
						)
		self.norms = nn.ModuleList([
							nn.RMSNorm(config.hidden_size) for _ in range(config.num_hidden_layers)
							])
		
		self.attentions = nn.ModuleList([
							EfficientDifferentialSkipAttention(
									hidden_size=config.hidden_size,
									kernel_size=config.kernel_size,
									forward_expansion=config.forward_expansion,
							) for _ in range(config.num_hidden_layers)
						])	
	
	
	def forward(self, input, mask):
		mask = make_score_mask(mask)

		x = self.embedder(input)
		for norm, attention in zip(self.norms, self.attentions):
			x = norm(x)	
			x = attention(x, mask)
		return x