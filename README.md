# deep_learning

## Train
``
python mnly.py
--do_train
--model_name_or_path albert-base-v2
--output_dir OUTPUT_DIR 
``

## Validation
``
python mnly.py 
--do_eval
--model_name_or_path albert-base-v2
--output_dir OUTPUT_DIR
``

## Test
``
python mnly.py
--do_prediction
--model_name_or_path albert-base-v2
--output_dir OUTPUT_DIR
``
