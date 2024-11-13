#COLORS
from colors import ATTRIBUTES, FOREGROUND_COLORS, RESET


#CUSTOM
from lib.configs import DataConfig, ModelConfig
from lib.utils import model_size, print_model_params, calculate_metrics, metrics_to_str

from lib.Models.final_classifiers import *
from lib.Models.models import *


dataset_config = DataConfig(
					dataset_name="TEST",
					dict_size=pow(2, 13),
					tokenizer_type="bpe",
					batch_size=128,
					max_len=256,
					labels=[0,1,2]
				)


model_config = ModelConfig( 
                    model_name="model_name", 
                    embedding_dimension=128,
                    reduced_embedding_dimension=16,
                    number_of_heads=None,
		            forward_expansion=1,
                    d_state=32,
                    num_layers=4,
                    max_length=dataset_config.max_len, 
                    vocab_size=dataset_config.dict_size,
                    learning_rate=1e-3,
                )
		

# model = BERT_original(model_config)
# model = Embedder_model(model_config)
# model = Embedder_conv_model(model_config)
# model = Mamba_model(model_config)
# model = Nano_Bert_Efficient(model_config)
# model = Nano_Bert_Efficient_mh(model_config)
# model = Embbert(model_config)
# model = Mamba_model_noNANO(model_config)
# model = Nano_Bert_Differential_Efficient(model_config)
model = Nano_Bert_Differential_Skip(model_config)

#cls = Classifier_max(model, model_config.embedding_dimension, dataset_config.n_labels())

print_model_params(model)

# print(cls.state_dict())