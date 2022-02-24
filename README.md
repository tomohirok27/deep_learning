# deep_learning

## Train
``
python mnli.py
--do_train
--model_name_or_path albert-base-v2
--output_dir OUTPUT_DIR 
``

## Validation
``
python mnli.py 
--do_eval
--model_name_or_path albert-base-v2
--output_dir OUTPUT_DIR
``

## Test
``
python mnli.py
--do_prediction
--model_name_or_path albert-base-v2
--output_dir OUTPUT_DIR
``
