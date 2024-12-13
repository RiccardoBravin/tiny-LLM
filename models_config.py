from lib.Models.BERT import BERT_Config
from lib.Models.mamba import MAMBA_Config
from lib.Models.NanoEmbedder import NanoEmbedder_Config
from lib.Models.NanoEmbedderConv import NanoEmbedderConv_Config
from lib.Models.NanoBERT import NanoBERT_Config
from lib.Models.BERTEfficient import BERTEfficient_Config
from lib.Models.NanoBERTEfficient import NanoBERTEfficient_Config
from lib.Models.EmbBERT import EmbBERT_Config


# MODEL CONFIGURATIONS

BERT_config = BERT_Config(
	vocab_size=pow(2,11),
	max_length=256,

	hidden_size=80,
	forward_expansion=2,
	num_attention_heads=2,
	num_hidden_layers=2,
    
	num_labels=2
)

MAMBA_config = MAMBA_Config(
	vocab_size=pow(2,11),
	max_length=256,

	hidden_size=64,
	forward_expansion=1,
    kernel_size=4,
    d_state=6,
	num_hidden_layers=5,
	
	num_labels=2
)

NanoEmbedder_config = NanoEmbedder_Config(
	vocab_size=pow(2,13),
	max_length=256,
	
	hidden_size=320,
	reduced_embedding=32,

	num_labels=2
)

NanoEmbedderConv_config = NanoEmbedderConv_Config(
    vocab_size=pow(2,13),
	max_length=256,
	
	hidden_size=320,
	reduced_embedding=32,
	kernel_size=16,

	num_labels=2
)

NanoBERT_config = NanoBERT_Config(
	vocab_size=pow(2,13),
	max_length=256,
	
	hidden_size=90,
	reduced_embedding=16,
	forward_expansion=2,
	num_attention_heads=2,
	num_hidden_layers=2,
	
	num_labels=2
)

BERTEfficient_config = BERTEfficient_Config(
	vocab_size=pow(2,11),
	max_length=256,
	
	hidden_size=84,
	forward_expansion=2,
	num_attention_heads=2,
	num_hidden_layers=3,
	
	num_labels=2
)

NanoBERTEfficient_config = NanoBERTEfficient_Config(
	vocab_size=pow(2,13),
	max_length=256,
	
	hidden_size=128,
	reduced_embedding=16,
	forward_expansion=0.7,
	num_attention_heads=1,
	num_hidden_layers=4,
	
	num_labels=2
)

EmbBERT_config = EmbBERT_Config(
	vocab_size=pow(2,13),
	max_length=256,
	
	hidden_size=128,
	reduced_embedding=16,
	forward_expansion=1,
	kernel_size=32,
	num_attention_heads=1,
	num_hidden_layers=4,
    
	num_labels=2
)

EmbBERT_BIG_config = EmbBERT_Config(
    variation="BIG",
	vocab_size=pow(2,13),
	max_length=512,
	
	hidden_size=128,
	reduced_embedding=32,
	forward_expansion=2,
	kernel_size=32,
	num_attention_heads=1,
	num_hidden_layers=5,
    
	num_labels=2
)