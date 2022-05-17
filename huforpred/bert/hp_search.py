import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import datasets
import pickle
from pathlib import Path
from datasets import load_metric
import numpy as np


def tune_transformer(
    data_dir,
    out_dir,
    num_samples=8,
    gpus_per_trial=0,
    smoke_test=False,
    num_labels=2,
    task_name="sentiment-analysis",
    model_name="SZTAKI-HLT/hubert-base-cc",
):

    labeled_data = datasets.Dataset.from_parquet(
        {
            "train": (data_dir / "labelled_train.parquet").as_posix(),
            "test": (data_dir / "labelled_test.parquet").as_posix(),
        },
        features=datasets.Features(
            {
                "text": datasets.Value("string"),
                "label": datasets.ClassLabel(num_classes=num_labels),
            }
        ),
    )

    config = AutoConfig.from_pretrained(
        model_name, num_labels=num_labels, finetuning_task=task_name
    )

    # Download and cache tokenizer, model, and features
    print("Downloading and caching Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Triggers tokenizer download to cache
    print("Downloading and caching pre-trained model")
    AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )

    def get_model():
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
        )

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
        output_dir=".",
        learning_rate=1e-5,  # config
        do_train=True,
        do_eval=True,
        no_cuda=gpus_per_trial <= 0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=2,  # config
        max_steps=-1,
        per_device_train_batch_size=16,  # config
        per_device_eval_batch_size=16,  # config
        warmup_steps=0,
        weight_decay=0.1,  # config
        logging_dir="./logs",
        skip_memory_metrics=True,
        report_to="none",
        auto_find_batch_size = True,
        fp16=True,
        disable_tqdm=True,
    )

    trainer = Trainer(
        model_init=get_model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        compute_metrics=compute_metrics,
        tokenizer = tokenizer,
        data_collator=data_collator,
    )

    tune_config = {
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 32,
        "num_train_epochs": tune.choice([2, 3, 4, 5]),
        "max_steps": 1 if smoke_test else -1,  # Used for smoke test.
    }

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="eval_acc",
        mode="max",
        perturbation_interval=1,
        hyperparam_mutations={
            "weight_decay": tune.uniform(0.0, 0.3),
            "learning_rate": tune.uniform(1e-5, 5e-5),
            "per_device_train_batch_size": [16, 32, 64],
        },
    )

    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "learning_rate": "lr",
            "per_device_train_batch_size": "train_bs/gpu",
            "num_train_epochs": "num_epochs",
        },
        metric_columns=["accuracy"],
    )

    best_run = trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        backend="ray",
        n_trials=num_samples,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        scheduler=scheduler,
        keep_checkpoints_num=1,
        checkpoint_score_attr="training_iteration",
        stop={"training_iteration": 1} if smoke_test else None,
        progress_reporter=reporter,
        local_dir="~/ray_results/",
        name="tune_transformer_pbt",
        log_to_file=True,
    )
    
    out_dir.write_bytes(pickle.dumps(best_run))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("datapath", type=Path)
    parser.add_argument("outpath", type=Path)

    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Address to use for Ray. "
        'Use "auto" for cluster. '
        "Defaults to None for local.",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using Ray Client.",
    )

    args, _ = parser.parse_known_args()

    if args.smoke_test:
        ray.init()
    else:
        ray.init(args.ray_address)

    if args.smoke_test:
        tune_transformer(
            data_dir=args.datapath,
            out_dir=args.outpath,
            num_samples=1,
            gpus_per_trial=0,
            smoke_test=True,
        )
    else:
        # You can change the number of GPUs here:
        tune_transformer(
            data_dir=args.datapath,
            out_dir=args.outpath,
            num_samples=8,
            gpus_per_trial=1,
        )
