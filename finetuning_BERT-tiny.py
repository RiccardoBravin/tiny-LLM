from datasets import load_dataset
from lib.preprocessing import dataset_selector
from colors import ATTRIBUTES, FOREGROUND_COLORS, RESET
from lib import utils

epochs_training = 3

# for DATASET_NAME in ["news", "bull", "limit", "nlu", "snips", "imdb", "emotion_split"]: #extra
# for DATASET_NAME in ["cola", "mrpc", "qnli", "qqp", "rte", "sst2", "wnli", "stsb",  "mnli-m", "mnli-mm"]: #GLUE
for DATASET_NAME in ["mnli-m", "mnli-mm"]: #GLUE
    #missing  = "stsb"  and "qnli"
# for DATASET_NAME in ["snips", "imdb", "emotion_split"]: #extra
    print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS['BrightYellow']}Loading dataset {DATASET_NAME} {RESET}")

    dataset_name = DATASET_NAME


    train_data, test_data = dataset_selector(dataset_name)

    num_labels = train_data.unique("label")
    if dataset_name == "stsb":
        num_labels = [0]


    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)


    train_dataset = train_data.map(tokenize_function, batched=True)
    eval_dataset = test_data.map(tokenize_function, batched=True)

    small_train_dataset = train_dataset.shuffle()#.select(range(1000))
    small_eval_dataset = eval_dataset.shuffle()#.select(range(1000))


    import numpy as np
    import evaluate
    from lib.utils import spearman_correlation

    accuracy_metric = evaluate.load("accuracy")
    mcc_metric = evaluate.load("matthews_correlation")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        try:
            predictions = np.argmax(logits, axis=-1)

            # Compute accuracy
            accuracy = accuracy_metric.compute(predictions=predictions, references=labels)

            # Compute MCC
            mcc = mcc_metric.compute(predictions=predictions, references=labels)

            # Compute F1 score
            f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')

            # Combine all metrics into a single dictionary
            return {
                "accuracy": accuracy["accuracy"],
                "matthews_correlation": mcc["matthews_correlation"],
                "f1": f1["f1"]
            }
        except:

            spe_corr = spearman_correlation(torch.tensor(labels), torch.tensor(logits))

            return {
                "spearman_correlation": spe_corr
            }

    for times in range(5):
        print(f"{ATTRIBUTES['Bold']}{FOREGROUND_COLORS['BrightCyan']}Training step: {times} {RESET}")
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=len(num_labels))

        print(utils.model_size(model))



        from transformers import TrainingArguments
        training_args = TrainingArguments("test_trainer", num_train_epochs=epochs_training, save_strategy="no", seed=times)




        from transformers import Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        res = trainer.evaluate()
        print(res)

        import os

        folder_name = f"BERT-tiny"
        if not os.path.exists(f"results/{folder_name}/"):
            os.makedirs(f"results/{folder_name}/")


        with open(f"results/BERT-tiny/{dataset_name}_report.txt", "a") as f:
                        f.write(f"{res}")
                        f.write("\n\n*******************************************\n\n")


