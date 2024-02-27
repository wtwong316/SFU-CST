#!/bin/bash
source venv/bin/activate
python webdemo.py --do_predict --overwrite_cache --prompt_column prompt --response_column response --model_name_or_path THUDM/chatglm2-6b --ptuning_checkpoint ./depression-chatglm2-6b/checkpoint-3000 --output_dir ./output --overwrite_output_dir --max_source_length 256 --max_target_length 256 --per_device_eval_batch_size 1 --predict_with_generate --pre_seq_len 128 --quantization_bit 8 --local_rank -1
