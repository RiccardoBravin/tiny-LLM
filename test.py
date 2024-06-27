# import torch
# import torch.nn.functional as F

# # from lib.utils import activations_calculator, print_model_params
# # from lib.configs import DataConfig, ModelConfig
# # from lib.Models.models import Mamba_model

# # dataset_config = DataConfig(
# #                     dataset_name="imdb", 
# #                     dict_size=pow(2, 12), 
# #                     tokenizer_type="wordpiece", #wordpiece #bpe #unigram 
# #                     batch_size=128, 
# #                     max_len=512, 
# #                     labels=None
# #                 )

# # model_config = ModelConfig(
# #                     model_name=None, 
# #                     embedding_dimension=64, 
# #                     reduced_embedding_dimension=16, 
# #                     number_of_heads=8, 
# #                     max_length=dataset_config.max_len, 
# #                     forward_expansion=3, 
# #                     num_layers=2,
# #                     vocab_size=dataset_config.dict_size
# #                 )

# # batch = 1


# # mamba = Mamba_model(model_config).to("cuda")

# # activations_calculator(mamba, dataset_config.dict_size, dataset_config.max_len)
# # print_model_params(mamba)

# # x = torch.randint(0, dataset_config.dict_size, (batch, dataset_config.max_len)).to("cuda")

# # y = mamba(x)

# # print(x.shape, y.shape)


# from lib.utils import n_ary_gray_code

# g_code = n_ary_gray_code(4, 3)
# g_code = torch.Tensor(g_code) -1 
# print(g_code)



# #Devo prendere il primo vettore e verificare che la cos sim sia > 0 per tutti gli altri  poi prendere il secondo e verificare che la cos sim sia > 0 per i vettori a destra e < 0 per i vettori a sinistra e così via
# # for i, vec in enumerate(g_code):
# #     for j, vec2 in enumerate(g_code):
# #         if i != j:
# #             print(f"{vec} vs {vec2}: {F.cosine_similarity(torch.Tensor(vec), torch.Tensor(vec2), dim=0).item()}")


# #mettere in ordine con la cosine similarity 
# for i, vec1 in enumerate(g_code):
#     for j, vec2 in enumerate(g_code):
#         if i != j:
#             #sort by cosine similarity
#             if F.cosine_similarity(vec1, vec2, dim=0).item() < 0:
#                 #swap vec1 and vec2 in g_code
#                 temp = g_code[i].clone()
#                 g_code[i] = g_code[j].clone()
#                 g_code[j] = temp



# #print cos sim of near vectors
# for i, vec in enumerate(g_code):
#     for j, vec2 in enumerate(g_code):
#         if i == j-1:
#             print(f"{vec} vs {vec2}: {F.cosine_similarity(vec, vec2, dim=0).item()}")


import torch
from lib.configs import *
from lib.Models.blocks import MamBravBlock
from lib.Models.models import MamBra_model

model_config = ModelConfig(
                    model_name=None, 
                    embedding_dimension=64, 
                    reduced_embedding_dimension=16, 
                    number_of_heads=8, 
                    max_length=512, 
                    forward_expansion=3, 
                    num_layers=2,
                    vocab_size=2**12
                )


model = MamBra_model(model_config)

x = torch.randint(0, model_config.vocab_size, (1, model_config.max_length))

y = model(x)

print(y)
print(y.shape)