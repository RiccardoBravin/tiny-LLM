import torch
from lib.Models.structures import Att_idea1
from lib.Models.models import New_idea1
from lib.configs import DataConfig, ModelConfig

from lib.Models.blocks import NewPosAttention

model_config = ModelConfig(
					model_name=None,
					embedding_dimension=128,
					reduced_embedding_dimension=16,
					number_of_heads=None,
					max_length=256,
					forward_expansion=0,
					d_state=0,
					num_layers=2,
					vocab_size=pow(2, 13),
					learning_rate = 2e-5,
				)

#model = New_idea1(model_config=model_config)
model = NewPosAttention(128, 256, 4)

print(model)

x = torch.randn(32, 256, 128)
mask = torch.ones(32, 256,256).bool()
out = model(x,x,x,mask)

print(out)
print(out.shape)