import torch

from lib.utils import activations_calculator, print_model_params
from lib.configs import DataConfig, ModelConfig
from lib.Models.models import Mamba_model

dataset_config = DataConfig(
                    dataset_name="imdb", 
                    dict_size=pow(2, 12), 
                    tokenizer_type="wordpiece", #wordpiece #bpe #unigram 
                    batch_size=128, 
                    max_len=512, 
                    labels=None
                )

model_config = ModelConfig(
                    model_name=None, 
                    embedding_dimension=64, 
                    reduced_embedding_dimension=16, 
                    number_of_heads=8, 
                    max_length=dataset_config.max_len, 
                    forward_expansion=3, 
                    num_layers=2,
                    vocab_size=dataset_config.dict_size
                )

batch = 1


mamba = Mamba_model(model_config).to("cuda")

activations_calculator(mamba, dataset_config.dict_size, dataset_config.max_len)
print_model_params(mamba)

x = torch.randint(0, dataset_config.dict_size, (batch, dataset_config.max_len)).to("cuda")

y = mamba(x)

print(x.shape, y.shape)


