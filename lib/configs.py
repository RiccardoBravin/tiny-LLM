from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str
    embedding_dimension: int
    reduced_embedding_dimension: int
    number_of_heads: int
    max_length: int
    forward_expansion: float
    num_layers: int
    vocab_size: int
    
    def feed_forward_hidden(self) -> int:
        return int(self.embedding_dimension * self.forward_expansion)

@dataclass
class DataConfig:
    dataset_name: str
    dict_size: int
    tokenizer_type: str
    batch_size: int
    max_len: int
    labels: list
    special_tokens: list | None = None
    mask_tok_id:int = 3
    pad_tok_id:int = 0
    def __post_init__(self):
        if self.tokenizer_type not in ["bpe", "wordpiece", "unigram"]:
            raise ValueError(f"The tokenizer type {self.tokenizer_type} is not supported, choose one of 'bpe', 'wordpiece', 'unigram'")
        if self.special_tokens is None:
            self.special_tokens = [0,1,2,3,4]
        
    def n_labels(self) -> int:
        return len(self.labels)
    