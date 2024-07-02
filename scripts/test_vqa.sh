export CUDA_VISIBLE_DEVICES=0

python -m test.test_answer_generator \
    --test_file /PATH/TO/TEST_FILE \
    --retrieval_results /PATH/TO/RETRIEVAL_RESULTS \
    --answer_generator /PATH/TO/ANSWER_GENERATOR \
    --llm_checkpoint /PATH/TO/LLM_CHECKPOINT \
    --output_file /PATH/TO/OUTPUT_FILE \
