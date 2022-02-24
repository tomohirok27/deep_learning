
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    raw_datasets = load_dataset("glue", 'mnli')
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=3,
        finetuning_task='mnli',
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    # A map from label to index for the model
    label_list = raw_datasets["train"].features["label"].names
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    # tokenizer funcation
    def preprocess_function(examples):

        args = examples['premise'], examples['hypothesis']
        result = tokenizer(*args, padding="max_length", max_length=data_args.max_seq_length, truncation=True)

        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )

    # get the metric 
    metric = load_metric("glue", "mnli")
    def compute_metrics(p: EvalPrediction):

        preds = p.predictions
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)

        return result

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation_matched"]
    predict_dataset = raw_datasets["test_matched"]

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # train
    if training_args.do_train:
        # training
        train_result = trainer.train()

        # evaluate train metrics
        metrics = train_result.metrics

        # save model
        trainer.save_model() 

        # log and save train metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        trainer.save_state()

    # evaluate
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # evaluate metrics with matched dataset (validation)

        eval_results = trainer.predict(eval_dataset)
        
        metrics = eval_results.metrics
        evaluation = eval_results.predictions
        evaluation =  np.argmax(evaluation, axis=1)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        output_predict_file = os.path.join(training_args.output_dir, "eval_results.txt")

        with open(output_predict_file, "w") as f1:
            f1.writelines(["pred  gold  premise  hypothesis  \n"])
            for pred, premise, hypothesis, gold in zip(evaluation, eval_dataset['premise'], eval_dataset['hypothesis'], eval_dataset['label']):
                label = label_list[pred]
                gold_label = label_list[gold]
                f1.writelines([label, "\t", gold_label, "\t", premise, "\t", hypothesis, "\n"])
        
        # evaluate metrics with miss matched dataset (validation)
        eval_dataset_mm = raw_datasets["validation_mismatched"]
        eval_results_mm = trainer.predict(eval_dataset_mm)
        metrics_mm = eval_results_mm.metrics
        evaluation_mm = eval_results_mm.predictions
        evaluation_mm =  np.argmax(evaluation_mm, axis=1)
        trainer.log_metrics("eval_mm", metrics_mm)
        trainer.save_metrics("eval_mm", metrics_mm)

        output_predict_file = os.path.join(training_args.output_dir, "eval_results_mm.txt")

        with open(output_predict_file, "w") as f2:
            f2.writelines(["pred  gold  premise  hypothesis  \n"])
            for pred, premise, hypothesis, gold in zip(evaluation_mm, eval_dataset_mm['premise'], eval_dataset_mm['hypothesis'], eval_dataset_mm['label']):
                label = label_list[pred]
                gold_label = label_list[gold]
                f2.writelines([label, "\t", gold_label, "\t", premise, "\t", hypothesis, "\n"])

    # predict 
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_dataset = predict_dataset.remove_columns("label")

        # predict with matched dataset (test)
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = predict_results.predictions
        predictions =  np.argmax(predictions, axis=1)

        output_predict_file = os.path.join(training_args.output_dir, "predict_results.txt")

        with open(output_predict_file, "w") as f3:
            for pred, premise, hypothesis in zip(predictions, predict_dataset['premise'], predict_dataset['hypothesis']):
                label = label_list[pred]
                f3.writelines([label, "\t", premise, "\t", hypothesis, "\n"])


        # predict with miss matched dataset (test)
        predict_dataset_mm = raw_datasets["test_mismatched"]
        predict_dataset_mm = predict_dataset_mm.remove_columns("label")
        predict_results_mm = trainer.predict(predict_dataset_mm, metric_key_prefix="predict_mm")
        predictions_mm = predict_results_mm.predictions
        predictions_mm =  np.argmax(predictions_mm, axis=1)

        output_predict_file_mm = os.path.join(training_args.output_dir, "predict_results_mm.txt")

        with open(output_predict_file_mm, "w") as f4:
            for pred, premise, hypothesis in zip(predictions_mm, predict_dataset_mm['premise'], predict_dataset_mm['hypothesis']):
                label = label_list[pred]
                f4.writelines([label, "\t", premise, "\t", hypothesis, "\n"])
  



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
