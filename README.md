# deep_learning

## Train
``
python mnly.py
--do_train
--task_name mnli
--model_name_or_path albert-base-v2
--output_dir OUTPUT_DIR 
``

## Validation
``
python mnly.py 
--do_eval
--task_name mnli
--model_name_or_path albert-base-v2
--output_dir OUTPUT_DIR
``

## Test
``
python mnly.py
--do_prediction
--task_name mnli
--model_name_or_path albert-base-v2
--output_dir OUTPUT_DIR
``
