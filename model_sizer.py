from lib.Models.BERT import BERT_Config
from lib.Models.mamba import MAMBA_Config
from lib.Models.NanoEmbedder import NanoEmbedder_Config
from lib.Models.NanoEmbedderConv import NanoEmbedderConv_Config
from lib.Models.NanoBERT import NanoBERT_Config
from lib.Models.BERTEfficient import BERTEfficient_Config
from lib.Models.NanoBERTEfficient import NanoBERTEfficient_Config
from lib.Models.EmbBERT import EmbBERT_Config

from lib.Models.classifiers import SequenceClassifier

from transformers import Trainer, TrainingArguments, BitsAndBytesConfig

from lib.utils import print_model_params

model_config = EmbBERT_Config(
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
		

model = SequenceClassifier(model_config)

print_model_params(model.model)




trainer = Trainer(
            model=model
    )

trainer.save_model(f"./TESTING/")

q_conf = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )

classifier = SequenceClassifier.from_pretrained(
                                f"./TESTING/",
                                config=model_config,
                                quantization_config = q_conf,
                            )


from peft import LoraConfig, get_peft_model
peft_config = LoraConfig(
    target_modules="all-linear",
)
classifier = get_peft_model(classifier, peft_config)
print(f"Model size: {classifier.get_memory_footprint()/1000}KB")
classifier.print_trainable_parameters()

