export CUDA_VISIBLE_DEVICES=6
TIME=`date +%m-%d-%H-%M`


python -m test.test_reranker \
    --test_file /PATH/TO/TESTFILE\
    --knowledge_base /PATH/TO/KNOWLEDGE_BASE_JSON_FILE\
    --faiss_index /PATH/TO/KNOWLEDGE_BASE_FAISS_INDEX\
    --retriever_vit eva-clip \
    --top_ks 1,5,10,20 \
    --retrieval_top_k 20\
    --perform_qformer_reranker \
    --qformer_ckpt_path /PATH/TO/QFORMER-CKPT\
    
 