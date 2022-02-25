# deep_learning

Fine-tune ALBERT by MNLI corpus

# Prepare
``
pip install transformers
``

## Train
``
python mnli.py
--do_train
--model_name_or_path albert-base-v2
--output_dir OUTPUT_DIR 
``

Other available options can be found in transformers.TrainingArguments

## Validation
``
python mnli.py 
--do_eval
--model_name_or_path albert-base-v2
--output_dir OUTPUT_DIR
``

Other available options can be found in transformers.TrainingArguments

## Test
``
python mnli.py
--do_prediction
--model_name_or_path albert-base-v2
--output_dir OUTPUT_DIR
``

Other available options can be found in transformers.TrainingArguments
