import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from lib.embedders import Nano_Embedder


from lib.utils import make_score_mask

class NanoEmbedderConv_Config(PretrainedConfig):
	model_type = "NanoEmbedderConv"
	def __init__(self, reduced_embedding:int=16, max_length:int=256, kernel_size:int=32, **kwargs):
		super().__init__(**kwargs)
		self.reduced_embedding = reduced_embedding
		self.max_length = max_length
		self.kernel_size = kernel_size



class NanoEmbedderConv(PreTrainedModel):
	config_class = NanoEmbedderConv_Config
	def __init__(self, config:NanoEmbedderConv_Config):
		super().__init__(config)
		self.embedder = Nano_Embedder(
								vocab_size=config.vocab_size, 
								max_len=config.max_length, 
								hidden_size=config.hidden_size, 
								reduced_embedding=config.reduced_embedding,
								segments=3
						)
		
	
		self.conv1d = nn.Conv1d(in_channels=config.hidden_size, out_channels=config.hidden_size, 
							  kernel_size=config.kernel_size,
							  groups=config.hidden_size,
							  padding=config.kernel_size - 1)
		
	
	
	def forward(self, input, mask=None):

		l = input.shape[1]

		# embedding the indexed sequence to sequence of vectors
		x = self.embedder(input)
		
		# applying the conv1d layer
		x = x.transpose(1, 2)
		x = self.conv1d(x)[:,:,:l]
		x = x.transpose(1, 2)

		return x