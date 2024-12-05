
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from lib.embedders import STD_Embedder


class BERT_Config(PretrainedConfig):
	model_type = "BERT"
	def __init__(self, forward_expansion:float=1, max_length:int=256, **kwargs):
		super().__init__(**kwargs)
		self.forward_expansion = forward_expansion
		self.max_length = max_length
		



class BERT(PreTrainedModel):
	config_class = BERT_Config
	def __init__(self, config:BERT_Config):
		super().__init__(config)

		
		self.embedder = STD_Embedder(
								vocab_size=config.vocab_size, 
								max_length=config.max_length, 
								hidden_size=config.hidden_size, 
								segments=3
						)
		

		encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, 
												   nhead=config.num_attention_heads,
												   dim_feedforward=int(config.forward_expansion*config.hidden_size),
												   batch_first=True,
						) 
 
		self.bert_layers = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)
	
	
	
	def forward(self, input, mask):
		# embedding the indexed sequence to sequence of vectors
		x = self.embedder(input)
		
		# applying the transformer layers
		x = self.bert_layers(x)

		return x