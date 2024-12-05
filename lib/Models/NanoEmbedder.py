
from transformers import PreTrainedModel, PretrainedConfig
from lib.embedders import Nano_Embedder


class NanoEmbedder_Config(PretrainedConfig):
	model_type = "NanoEmbedder"
	def __init__(self, reduced_embedding:int=16, max_length:int=256, **kwargs):
		super().__init__(**kwargs)
		self.reduced_embedding = reduced_embedding
		self.max_length = max_length
		



class NanoEmbedder(PreTrainedModel):
	config_class = NanoEmbedder_Config
	def __init__(self, config:NanoEmbedder_Config):
		super().__init__(config)
		self.embedder = Nano_Embedder(
								vocab_size=config.vocab_size, 
								max_len=config.max_length, 
								hidden_size=config.hidden_size, 
								reduced_embedding=config.reduced_embedding,
								segments=3
						)
		
	
	
	def forward(self, input, mask=None):

		x = self.embedder(input)

		return x