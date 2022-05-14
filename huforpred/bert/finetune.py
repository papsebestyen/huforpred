from transformers import (
    BertForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import datasets
from datasets import load_metric
import numpy as np
from pathlib import Path
import os

training_data_path = Path("../../data/training")

labeled_data = datasets.Dataset.from_parquet(
    {
        "train": (training_data_path / "labelled_train.parquet").as_posix(),
        "test": (training_data_path / "labelled_test.parquet").as_posix(),
    },
    features=datasets.Features(
        {"text": datasets.Value("string"), "label": datasets.ClassLabel(num_classes=2)}
    ),
)

hugging_face_model = "NYTK/sentiment-hts2-hubert-hungarian"
tokenizer = AutoTokenizer.from_pretrained(hugging_face_model)
model = BertForSequenceClassification.from_pretrained(hugging_face_model, num_labels=2)


def preprocess_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=512
    )


tokenized_data = labeled_data.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = metric.compute(predictions=predictions, references=labels)
    return acc


training_args = TrainingArguments(
    'papsebestyen/fin-hubert',
    # output_dir="./results2",
    learning_rate=2e-6,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    dataloader_num_workers=40,
    no_cuda=True,
    hub_model_id  = 'papsebestyen/fin-hubert',
    push_to_hub=True,
    hub_token = os.environ['HUGGINGFACE_TOKEN'],
    hub_private_repo =True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"].shuffle(),
    eval_dataset=tokenized_data["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
