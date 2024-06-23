export CUDA_VISIBLE_DEVICES=0

# torchrun --nnodes=1 --nproc_per_node=4  fine_tune_qformer_dp.py \
python fine_tune_qformer.py \
   --blip-model-name 'blip2_reranker' \
   --num-epochs 20 \
   --num-workers 4 \
   --learning-rate 1e-5 \
   --batch-size 6 \
   --transform targetpad \
   --target-ratio 1.25  \
   --save_frequency 50000 \
   --train_file /PATH/TO/TRAIN_FILE \
   --knowledge_base_file /PATH/TO/KNOWLEDGE_BASE_JSON_FILE \
   --negative_db_file /PATH/TO/NEGATIVE_DB_FILE \
   --inat_id2name /PATH/TO/INATURALISTS_ID2NAME \
   --save-training \
   --validation-frequency 1 \
   # --world-size 4 \ 