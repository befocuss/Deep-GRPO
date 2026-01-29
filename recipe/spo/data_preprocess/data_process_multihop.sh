LOCAL_DIR=/data/hf-datasets/hotpotqa

## process multiple dataset search format train file
DATA=hotpotqa
python recipe/spo/data_preprocess/hotpotqa_train.py --local_dir $LOCAL_DIR --data_sources $DATA

## process multiple dataset search format test file
DATA=hotpotqa
python recipe/spo/data_preprocess/hotpotqa_test.py --local_dir $LOCAL_DIR --data_sources $DATA
