# EchoSight: Advancing Visual-Language Models with Wiki Knowledge
This is official repository of EchoSight: Advancing Visual-Language Models with Wiki Knowledge
<img width="1728" alt="image" src="https://github.com/Go2Heart/EchoSight/assets/71871209/c9eb99d2-fc90-469f-ac69-5f28093863ab">

<img width="1728" alt="image" src="https://github.com/Go2Heart/EchoSight/assets/71871209/4257097c-b7d0-4c8f-9436-b5f18253bd13">






## Requirements
1. (Optional) Creating conda environment

```bash
conda create -n echosight python=3.10
conda activate echosight
```

2. Install the required packages
```bash
pip install -r requirements.txt
```
## Training
The multimodal reranker of EchoSight is trained using Encyclopedic-VQA datasets and the corresponding 2M Knowledge Base.

To train the multimodal reranker, run the bash script after changing the necessary configurations.
```bash
bash train_reranker.sh
```
**Script Details**
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

## Demo

## Citation

## Acknowledgements

## Release TODOs
1. update knowledge base & faiss index
2. infoseek training and testing file in EVQA-format
3. inaturalists id2name


