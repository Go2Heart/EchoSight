import os
import json,pickle,csv
import random
import PIL
import torch
from .dataset_utils import prepare_multimodal_doc, prepare_multimodal_query, reconstruct_wiki_article, get_image, reconstruct_wiki_sections
import time, re

class QFormerRerankerDataset(torch.utils.data.Dataset):
    """_summary_
        Dataset for training to retrieve semantically similar text.
        Used for training the QFormer model.
        """
    def __init__(
        self, 
        knowledge_base_file, 
        train_file,
        preprocess: callable,
        get_image_function=get_image,
        negative_db_file=None,
        retriever=None,
        visual_attr_file=None,
        use_negative=False,
        neg_num=4,
        inat_id2name=None
        ):
        # load the knowledge base
        with open(knowledge_base_file, "r") as f:
            self.knowledge_base = json.load(f)
        self.kb_keys = list(self.knowledge_base.keys())
        self.train_list = []
        self.url_list = []
        with open(train_file, "r") as f:
            reader = csv.reader(f)
            self.header = next(reader)
            # url_records = []
            for row in reader:
                if (row[self.header.index("question_type")] == "automatic" or row[self.header.index("question_type")] == "templated" or row[self.header.index("question_type")] == "multi_answer" or row[self.header.index("question_type")] == "infoseek"):
                    self.url_list.append(row[self.header.index("wikipedia_url")])
                    self.train_list.append(row)
        
        self.retriever = retriever
        self.preprocess = preprocess
        self.get_image = get_image_function
        self.max_length = 512
        
        if negative_db_file is not None:
            with open(negative_db_file, "r") as f:
                self.negative_db = json.load(f)
        else:
            self.negative_db = None
            
        if visual_attr_file is not None:
            with open(visual_attr_file, "r") as f:
                self.visual_attr = json.load(f)
        else:
            self.visual_attr = None
            
        self.use_negative = use_negative
        self.neg_num = neg_num
        if inat_id2name is not None:
            with open(inat_id2name, "r") as f:
                self.iNat_id2name = json.load(f)
        else:
            self.iNat_id2name = None
            
        
    def __len__(self):
        return len(self.train_list)
    
    def get_url_list(self):
        return self.url_list
    
    def __getitem__(self, idx):
        example = self.train_list[idx]
        question = example[self.header.index("question")]
        question_images = [self.get_image(image_id, example[self.header.index("dataset_name")], self.iNat_id2name) for image_id in example[self.header.index("dataset_image_ids")].split("|")]
        question_image_path = question_images[0]
        question_image = self.preprocess(PIL.Image.open(question_image_path))
        
        positive_url = example[self.header.index("wikipedia_url")]
        positive_entry = self.knowledge_base[positive_url]
        evidence_section_id = example[self.header.index("evidence_section_id")]
        positive_section, negative_sections = reconstruct_wiki_sections(positive_entry, evidence_section_id)
        
        if self.use_negative:
            if self.negative_db is not None:
                # choose from the hard negative samples
                neg_info = self.negative_db[question_image_path.split("/")[-1].split(".")[0]]
                neg_list = [entry["entry"] for entry in neg_info]
                sim_list = [entry["similarity"] for entry in neg_info]
                if positive_url in neg_list:
                    neg_list.remove(positive_url)
                negative_entry_keys = neg_list 
            else:
                # choose from knowledge base without the positive entry
                # create the keys list without the positive entry
                negative_entry_keys = random.choices(self.kb_keys, k=20)
                if positive_url in negative_entry_keys:
                    negative_entry_keys.remove(positive_url)
            negative_entries = [self.knowledge_base[key] for key in negative_entry_keys]
            
            for it,entry in enumerate(negative_entries):
                negative_section = reconstruct_wiki_sections(entry)
                negative_sections.extend(negative_section)
            random_index = random.sample(range(len(negative_sections)), self.neg_num*2)
            negative_sections = [negative_sections[i] for i in random_index]
        else:
            negative_sections = []
        return question_image, question, positive_section, negative_sections, positive_url