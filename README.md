# EchoSight: Advancing Visual-Language Models with Wiki Knowledge (EMNLP 2024 Findings)
This is the official PyTorch implementation of EchoSight: Advancing Visual-Language Models with Wiki Knowledge.

[[Project Page]](https://go2heart.github.io/echosight) [[Paper]](https://arxiv.org/abs/2407.12735)

<img width="1728" alt="image" src="https://github.com/Go2Heart/EchoSight/assets/71871209/c9eb99d2-fc90-469f-ac69-5f28093863ab">

<img width="1728" alt="image" src="https://github.com/Go2Heart/EchoSight/assets/71871209/4257097c-b7d0-4c8f-9436-b5f18253bd13">






## Requirements
1. (Optional) Create conda environment

```bash
conda create -n echosight python=3.10
conda activate echosight
```

2. Install the required packages
```bash
pip install -r requirements.txt
```
## Knowledge Base
We provide the knowledge bases used in EchoSight. The knowledge base file is the same format as the Encyclopedic-VQA dataset. Apart from the original 2M knowledge base for Encyclopedic-VQA, we also provide a 100K knowledge base for InfoSeek, which is a filtered subset of the 2M knowledge base. The knowledge base files can be downloaded from the following links:
### Enclyclopedic-VQA
- [Encylopedic-VQA's 2M Knowledge Base](https://storage.googleapis.com/encyclopedic-vqa/encyclopedic_kb_wiki.zip)
- [Enclopedic-VQA KB Images Faiss Index](https://drive.google.com/file/d/1cQYul-my2FtqfCND2FeqgF9TCMU3u5xz/view?usp=drive_link)
### Infoseek
- [Our InfoSeek's 100K Knowledge Base ](https://drive.google.com/file/d/1cIbKtYryD7XBAw0tjrrCvMCJC2rIzLM5/view?usp=drive_link)
- [InfoSeek KB Images Faiss Index](https://drive.google.com/file/d/1cDuL45c1iYwB0_BSlTmrMzbEE8ik2cVJ/view?usp=drive_link)

## VQA Questions
### Encyclopedic VQA
The VQA questions can be downloaded in .csv format here(Provided by Encyclopedic-VQA):

*   [train.csv](https://storage.googleapis.com/encyclopedic-vqa/train.csv)
*   [val.csv](https://storage.googleapis.com/encyclopedic-vqa/val.csv)
*   [test.csv](https://storage.googleapis.com/encyclopedic-vqa/test.csv)

To download the images in Encyclopedic-VQA:

- [iNaturalist 2021](https://github.com/visipedia/inat_comp/tree/master/2021)(Also put id2name file in the same folder, which can be downloaded from [train_id2name](https://drive.google.com/file/d/1cUP0sWtI4z7whH9V5FOvqfJ0LTxZLOd9/view?usp=drive_link) and [val_id2name](https://drive.google.com/file/d/1cYzo4qewPABFuoMhpME4j2DWAA_Y-l2L/view?usp=drive_link))

- [Google Landmarks Dataset V2](https://github.com/cvdfoundation/google-landmark)

### InfoSeek
The VQA questions of InfoSeek are transformed to E-VQA format from the original InfoSeek dataset. Due to the The questions can be downloaded in .csv format here:
* [train.csv](https://drive.google.com/file/d/1cQiQmdFq8_8gsaZPsmzKzIjZhdcd_kxP/view?usp=drive_link)
* [test.csv](https://drive.google.com/file/d/1cSG_dVuao9lKZy8vaUDWEo7mIHowjUeE/view?usp=drive_link)

To download the images in InfoSeek:

- [Oven](https://github.com/edchengg/oven_eval/tree/main/image_downloads)

## Training
The multimodal reranker of EchoSight is trained using Encyclopedic-VQA datasets and the corresponding 2M Knowledge Base. If you want to enable Hard Negative Sampling when training the reranker, we provide our Hard_Neg result sampled by Eva-CLIP here:
- [Hard Negative Sampling File](https://drive.google.com/file/d/1i8AzqyqG_QH0wCFHXgZzjt7tLbLWz616/view?usp=drive_link)


To train the multimodal reranker, run the bash script after changing the necessary configurations.
```bash
bash scripts/train_reranker.sh
```
### Script Details
The train_reranker.sh script is used to fine-tune the reranker module with specific parameters:

--`blip-model-name`: Name of the BLIP model to be used for reranking.

--`num-epochs`: Number of epochs for training. In this case, the model will be trained for 20 epochs.

--`num-workers`: Number of worker threads for data loading.

--`learning-rate`: Learning rate for the optimizer.

--`batch-size`: Number of samples per batch during training.

--`transform`: Transformation applied to the data. targetpad ensures the data is padded to a target size.

--`target-ratio`: Target aspect ratio for the padding transformation.

--`save_frequency`: Frequency (in steps) to save the model checkpoints. 

--`train_file`: Path to the training data file. The training file should be the same format as provided by Encyclopedic-VQA.

--`knowledge_base_file`: Path to the knowledge base file in JSON format. The format should be the same with that of the Encyclopedic-VQA.

--`negative_db_file`: Path to the hard negative sampled database file used for training.

--`inat_id2name`: Path to the iNaturalist ID to name mapping file.

--`save-training`: Flag to save the training progress.
## Inference
0. Our reranker module weights can be downloaded at [[Checkpoint]](https://drive.google.com/file/d/1d6QOyePuvHLlYxC1_Dvxvij3wNm3gOB9/view?usp=sharing).

2. To perform inference with the trained model, run the provided test_reranker.sh script after adjusting the necessary parameters.
```bash
bash scripts/test_reranker.sh
```
### Script Details
The test_reranker.sh script uses the following parameters for inference:

--`test_file`: Path to the test file.

--`knowledge_base`: Path to the knowledge base JSON file.

--`faiss_index`: Path to the FAISS index file for efficient similarity search.

--`retriever_vit`: Name of the visual transformer model used for initial retrieval. In the example script, eva-clip is used.

--`top_ks`: Comma-separated list of top-k recall results for retrieval (e.g., 1,5,10,20).

--`retrieval_top_k`: The top-k value used for retrieval.

--`perform_qformer_reranker`: Flag to perform reranking using QFormer.

--`qformer_ckpt_path`: Path to the QFormer checkpoint file.

--`perform_qformer_reranker`: Flag to perform the ultimate VQA.

--`save_result`: Flag to save the inference result.

--`save_result_path`: Path to the result json file would be saved.

--`resume_from`: Path to the retrieval result. If this parameter is used, the inference process will load the saved retrieval result, instead of using the retriever on-the-fly.

2. (Optional) With the saved retrieval or reranked results, an answer generation can be performed standalone.
```bash
bash scripts/test_vqa.sh
```
3. (Optional) Run the batch inference vqa script (Releasing Soon).
### Script Details
The test_vqa.sh script uses the following parameters for inference:

--`test_file`: Path to the test file.

--`retrieval_results`: Path to the retrieval result file.

--`answer_generator`: Name of the answer generator model to be used. Choose from [Mistral, LLaMA3, GPT4, PaLM].

--`llm_checkpoint`: Path to the [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) or [LLaMA3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) checkpoint file. If using GPT4 or PaLM, this parameter is not needed. Instead, change api_key in model/anwser_generator.py.

--`output_file`: Path to the output file. Default is ./answer.json.
## Demo
Run the demo of EchoSight.
```bash
python app.py
```
### Demo Showcase
<img width="1267" alt="image" src="https://github.com/Go2Heart/EchoSight/assets/71871209/7daa8ee7-5b3d-4789-ba43-16c481471b77">

<img width="1267" alt="image" src="https://github.com/Go2Heart/EchoSight/assets/71871209/77d04711-93fc-4944-a465-92ebe643d4b5">

## Citation
```
@misc{yan2024echosightadvancingvisuallanguagemodels,
      title={EchoSight: Advancing Visual-Language Models with Wiki Knowledge}, 
      author={Yibin Yan and Weidi Xie},
      year={2024},
      eprint={2407.12735},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.12735}, 
}
```
## Acknowledgements
Thanks to the code of [LAVIS](https://github.com/salesforce/LAVIS/tree/main) and data of [Encyclopedic-VQA](https://github.com/google-research/google-research/tree/master/encyclopedic_vqa) and [InfoSeek](https://github.com/open-vision-language/infoseek).


