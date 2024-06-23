"""  Serves as the retriever for the EchoSight.
"""

import torch
import tqdm
import pickle
import json
from transformers import AutoModel, AutoProcessor, CLIPVisionModel, CLIPImageProcessor
import faiss
import numpy as np
from faiss import write_index, read_index
import faiss.contrib.torch_utils
class KnowledgeBase:
    """ Knowledge base for EchoSight system.

    Returns:
        KnowledgeBase
    """
    
    def __len__(self):
        """Return the length of the knowledge base.

        Args:

        Returns:
            int
        """
        return len(self.knowledge_base)
    
    def __getitem__(self, index):
        """Return the knowledge base entry at the given index.

        Args:
            index (int): The index of the knowledge base entry to return.

        Returns:
            KnowledgeBaseEntry
        """
        return self.knowledge_base[index]
    
    def __init__(self, knowledge_base_path):
        """Initialize the KnowledgeBase class.

        Args:
            knowledge_base_path (str): The path to the knowledge base.

        Returns:
            KnowledgeBase
        """
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = None
    
    def load_knowledge_base(self):
        """Load the knowledge base.

        Args:

        Returns:
            KnowledgeBase
        """
        raise NotImplementedError
    
class WikipediaKnowledgeBase(KnowledgeBase):
    """ Knowledge base for EchoSight.

    Returns:
        KnowledgeBase
    """
    def __init__(self, knowledge_base_path):
        """Initialize the KnowledgeBase class.

        Args:
            knowledge_base_path (str): The path to the knowledge base.

        Returns:
            KnowledgeBase
        """
        super().__init__(knowledge_base_path)
        self.knowledge_base = []
    def load_knowledge_base_full(self, image_dict = None, scores_path = None, visual_attr = None):
        """Load the knowledge base.

        Args:
            knowledge_base: The knowledge base to load.
        """
        try:
            with open(self.knowledge_base_path, "rb") as f:
                knowledge_base_dict = json.load(f)
        except:
            raise FileNotFoundError("Knowledge base not found, which should be a url or path to a json file.")
        # image_dict and load_scores_path can't be both None and can't be both not None
                
        if visual_attr is not None:
            try:
                with open(visual_attr, 'r') as f:
                    visual_attr_dict = json.load(f)
            except:
                raise FileNotFoundError("Visual Attr not found, which should be a url or path to a json file.")
            
        if scores_path is not None:
            # get the image scores for each entry
            # get all the *.pkl files in the scores_path
            print("Loading knowledge base score from {}.".format(scores_path))
            import glob
            score_files = glob.glob(scores_path + "/*.pkl")
            image_scores = {}
            for score_file in tqdm.tqdm(score_files):
                try:
                    with open(score_file, "rb") as f:
                        image_scores.update(pickle.load(f))
                except:
                    raise FileNotFoundError("Image scores not found, which should be a url or path to a pickle file.")
            print("Loaded {} image scores.".format(len(image_scores)))
            for wiki_url, entry in knowledge_base_dict.items():
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                for url in wiki_entry.image_urls:
                    if url in image_scores:
                        wiki_entry.score[url] = image_scores[url]
                self.knowledge_base.append(wiki_entry)
        else:
            print("Loading knowledge base without image scores.") 
            for wiki_url, entry in knowledge_base_dict.items():
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                self.knowledge_base.append(wiki_entry)   

        print ("Loaded knowledge base with {} entries.".format(len(self.knowledge_base)))              
        return self.knowledge_base
    def load_knowledge_base(self, image_dict = None, scores_path = None, visual_attr = None):
        """Load the knowledge base.

        Args:
            knowledge_base: The knowledge base to load.
        """
        try:
            with open(self.knowledge_base_path, "rb") as f:
                knowledge_base_dict = json.load(f)
        except:
            raise FileNotFoundError("Knowledge base not found, which should be a url or path to a json file.")
        # image_dict and load_scores_path can't be both None and can't be both not None
        if visual_attr is not None:
            # raise NotImplementedError
            try:
                with open(visual_attr, 'r') as f:
                    visual_attr_dict = json.load(f)
            except:
                raise FileNotFoundError("Visual Attr not found, which should be a url or path to a json file.")
            
        if scores_path is not None: #TODO: fix the knowledge base and visual_attr is None:
            # get the image scores for each entry
            print("Loading knowledge base score from {}.".format(scores_path))
            try:
                with open(scores_path, "rb") as f:
                    image_scores = pickle.load(f)
            except:
                raise FileNotFoundError("Image scores not found, which should be a url or path to a pickle file.")
            for wiki_url, entry in knowledge_base_dict.items():
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                for url in wiki_entry.image_urls:
                    if url in image_scores:
                        wiki_entry.score[url] = image_scores[url]
                self.knowledge_base.append(wiki_entry)
        else:
            print("Loading knowledge base without image scores.") 
            for wiki_url, entry in knowledge_base_dict.items():
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                self.knowledge_base.append(wiki_entry)

        print ("Loaded knowledge base with {} entries.".format(len(self.knowledge_base)))              
        return self.knowledge_base
class WITKnowledgeBaseEntry:
    """ Knowledge base entry for EchoSight.

    Returns:
    """
    def __init__(self, title,entry_dict, visual_attr = None):
        """Initialize the KnowledgeBaseEntry class.

        Args:
            title (str): The title of the entry.
            entry_dict (dict): The dictionary containing the knowledge base entry.
            dict_keys are: "image_reference_descriptions","image_section_indices","image_urls","section_texts","section_titles","title","url"

        Returns:
            KnowledgeBaseEntry
        """
        self.title = entry_dict["title"]
        self.url = entry_dict["url"]
        self.image_urls = entry_dict["image_urls"]
        self.image_reference_descriptions = entry_dict["image_reference_descriptions"]
        self.image_section_indices = entry_dict["image_section_indices"]
        self.section_titles = entry_dict["section_titles"]
        self.section_texts = entry_dict["section_texts"]
        self.image = {}
        self.score = {}
        self.visual_attr = visual_attr
class WikipediaKnowledgeBaseEntry:
    """ Knowledge base entry for EchoSight.

    Returns:
    """
    def __init__(self, entry_dict, visual_attr = None):
        """Initialize the KnowledgeBaseEntry class.

        Args:
            entry_dict (dict): The dictionary containing the knowledge base entry.
            dict_keys are: "image_reference_descriptions","image_section_indices","image_urls","section_texts","section_titles","title","url"

        Returns:
            KnowledgeBaseEntry
        """
        self.title = entry_dict["title"]
        self.url = entry_dict["url"]
        self.image_urls = entry_dict["image_urls"]
        self.image_reference_descriptions = entry_dict["image_reference_descriptions"]
        self.image_section_indices = entry_dict["image_section_indices"]
        self.section_titles = entry_dict["section_titles"]
        self.section_texts = entry_dict["section_texts"]
        self.image = {}
        self.score = {}
        self.visual_attr = visual_attr
    def toDict(self):
        return {
            "title": self.title,
            "url": self.url,
            "image_urls": self.image_urls,
            "image_reference_descriptions": self.image_reference_descriptions,
            "image_section_indices": self.image_section_indices,
            "section_titles": self.section_titles,
            "section_texts": self.section_texts,
        }
        
class Retriever:
    """Retriever parent class for EchoSight.

    Attributes:
        model: The model to use for retrieval.
        device: The device to use for retrieval.
    """
    
    def __init__(self, model = None):
        """Initialize the Retriever class.

        Args:
            model: The model to use for retrieval.
            device: The device to use for retrieval.
            knowledge_base: The knowledge base to use for retrieval.
        """
        self.model = model
        
    def load_knowledge_base(self, knowledge_base_path):
        """Load the knowledge base.

        Args:
            knowledge_base_path: The knowledge base to load.
        """
        raise NotImplementedError

    def retrieve_image(self, image):
        """Retrieve the image.

        Args:
            image: The image to retrieve.

        Returns:
            The retrieved image.
        """
        raise NotImplementedError
    
class ClipRetriever(Retriever):
    """Image Retriever with CLIP-based VIT.

    Attributes:
        model: The model to use for retrieval.
        device: The device to use for retrieval.
    """
    
    def __init__(self, model = "clip", device = "cpu"):
        """Initialize the DummyRetriever class.

        Args:
            model: The model to use for retrieval.
            device: The device to use for retrieval.
            knowledge_base: The knowledge base to use for retrieval.
        """
        super().__init__(model)
        self.model_type = model
        if model == "clip":
            self.model = AutoModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=torch.float16,
                )
            del self.model.text_projection
            del self.model.text_model # avoiding OOM
            self.model.to('cuda').eval()
            self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        elif model == "eva-clip":
            self.model = AutoModel.from_pretrained(
                "BAAI/EVA-CLIP-8B", # TODO change into config
                torch_dtype=torch.float16,
                trust_remote_code=True)
            del self.model.text_projection 
            del self.model.text_model # avoiding OOM
            self.model.to('cuda').eval()
            self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.device = device
        self.model.to(device)
        self.knowledge_base = None
        
    def load_knowledge_base(self, knowledge_base_path, image_dict = None, scores_path = None, visual_attr = None):
        """Load the knowledge base.

        Args:
            knowledge_base_path: The knowledge base to load.
        """
        self.knowledge_base = WikipediaKnowledgeBase(knowledge_base_path)
        self.knowledge_base.load_knowledge_base(image_dict=image_dict, scores_path=scores_path, visual_attr=visual_attr)
        # if scores_path is a folder, then load all the scores in the folder, otherwise, load the single score file
    
    def save_knowledge_base_faiss(self, knowledge_base_path, image_dict = None, scores_path = None, visual_attr = None, save_path = None):
        self.knowledge_base = WikipediaKnowledgeBase(knowledge_base_path)
        if scores_path[-4:] == ".pkl":
            print("Loading knowledge base from {}.".format(scores_path))
            self.knowledge_base.load_knowledge_base(image_dict=image_dict, scores_path=scores_path, visual_attr=visual_attr)
        else:
            print("Loading full knowledge base from {}.".format(scores_path))
            self.knowledge_base.load_knowledge_base_full(image_dict=image_dict, scores_path=scores_path, visual_attr=visual_attr)
        self.prepare_faiss_index()
        self.save_faiss_index(save_path)
        
            
        
    
    def retrieve_image(self, image, top_k=100, pool_method="max", return_entry_list=False):
        """Retrieve the top K similar images from the knowledge base.

        Args:
            image: The image to be compared.
            top_k (int): The number of top similar images to retrieve.
            pool_method (str): The pooling method to use ('max' or 'average').

        Returns:
            list: Top k entries, every entry is a dict of (url, kb_index, similarity)
        """ 
        # Process the input image
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        inputs.to(self.device)
        outputs = self.model(**inputs)
        image_score = outputs.pooler_output
        # get the top k images in kb by cosine similarity
        kb_image_similarities = {}
        for i in range(len(self.knowledge_base)):
            kb_image_similarity = []
            wiki_url = self.knowledge_base[i].url
            image_urls = list(self.knowledge_base[i].score.keys())
            scores = [torch.tensor(self.knowledge_base[i].score[url]).to(self.device) for url in image_urls]
            if len(scores) == 0:
                continue
            scores_matrix = torch.stack(scores, dim=0)
            kb_image_similarity = torch.cosine_similarity(image_score.unsqueeze(0), scores_matrix, dim=-1).squeeze(0)
            if pool_method == "max":
                # get the max score
                # kb_image_similarity = torch.max(kb_image_similarity, dim=0)[0]
                # get the max score's index in the url list
                max_similarity_index = torch.argmax(kb_image_similarity, dim=0)
                max_similarity = kb_image_similarity[max_similarity_index]
                max_similarity_url = image_urls[max_similarity_index]
            else:
                raise NotImplementedError("Only max pooling is implemented.")
            # add key to the dict
            if wiki_url not in kb_image_similarities:
                kb_image_similarities[wiki_url] = {}
            kb_image_similarities[wiki_url].update({"similarity": max_similarity.item()})
            kb_image_similarities[wiki_url].update({"knowledge_base_index": i})
            kb_image_similarities[wiki_url].update({"image_url": max_similarity_url}) # TODO bug to fix, if multiple images of same entry are hit
            
            
        ranked_list = sorted(kb_image_similarities.items(), key=lambda x: x[1]["similarity"], reverse=True)
        # get the top k images' urls
        top_k_entries = []
        if return_entry_list:
            for i in range(top_k):
                top_k_entries.append(self.knowledge_base[ranked_list[i][1]["knowledge_base_index"]])
            return top_k_entries
        for i in range(top_k):
            top_k_entries.append({"url": ranked_list[i][0], "knowledge_base_index": ranked_list[i][1]["knowledge_base_index"], "image_url": ranked_list[i][1]["image_url"],"similarity": ranked_list[i][1]["similarity"], "kb_entry": self.knowledge_base[ranked_list[i][1]["knowledge_base_index"]]})
        
        return top_k_entries
    
    def save_faiss_index(self, save_index_path):
        if save_index_path is not None:
            write_index(self.faiss_index, save_index_path+"kb_index.faiss")
            with open(save_index_path+"kb_index_ids.pkl", "wb") as f:
                pickle.dump(self.faiss_index_ids, f)
            
    def load_faiss_index(self, load_index_path):
        if load_index_path is not None:
            self.faiss_index = read_index(load_index_path+"kb_index.faiss")
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
            with open(load_index_path+"kb_index_ids.pkl", "rb") as f:
                self.faiss_index_ids = pickle.load(f)
                
            print ("Faiss index loaded with {} entries.".format(self.faiss_index.ntotal))
        return
    
    def prepare_faiss_index(self, save_index_path = None):
        # use the knowledge base's score element to build the index
        # get the image scores for each entry
        scores = [score for entry in self.knowledge_base for score in entry.score.values()]
        score_ids = [i for i in range(len(self.knowledge_base)) for j in range(len(self.knowledge_base[i].score))]
        # import ipdb; ipdb.set_trace()
        index = faiss.IndexFlatIP(scores[0].shape[0])
        # res = faiss.StandardGpuResources()
        # index = faiss.index_cpu_to_gpu(res, 0, index)
        np_scores = np.array(scores)
        np_scores = np_scores.astype(np.float32)
        faiss.normalize_L2(np_scores)
        index.add(np_scores)
        self.faiss_index = index
        self.faiss_index_ids = score_ids
        print ("Faiss index built with {} entries.".format(index.ntotal))

        return         
    
    @torch.no_grad()
    def retrieve_image_faiss(self, image, top_k=100, pool_method="max", return_entry_list=False):
        """Retrieve the top K similar images from the knowledge base using faiss.

        Args:
            image: The image to be compared.
            top_k (int): The number of top similar images to retrieve.
            pool_method (str): The pooling method to use ('max' or 'average').

        Returns:
            list: Top k entries, every entry is a dict of (url, kb_index, similarity)
        """ 
        if self.model_type == "clip":
            # CLIP Process the input image
            # inputs = self.processor(images=image, return_tensors="pt", padding=True)
            # inputs.to(self.device)
            # outputs = self.model(**inputs)
            # image_score = outputs.pooler_output
            inputs = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device).half()
            image_score = self.model.get_image_features(inputs)
        elif self.model_type == "eva-clip":
            # EVA-CLIP Process the input image
            inputs = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device).half()
            image_score = self.model.encode_image(inputs)
        assert self.faiss_index and self.faiss_index_ids is not None
        # kb_image_similarities = {}
        query = image_score.float()
        query = torch.nn.functional.normalize(query)
        # faiss.normalize_L2(query)
        D,I = self.faiss_index.search(query, top_k)
        top_k_entries = []
        for i in range(top_k): # for each image in the top k
            if return_entry_list:
                top_k_entries.append(self.knowledge_base[self.faiss_index_ids[I[0][i]]])
            else:
                # find the first knowledge base entry that contains the image
                index_id = self.faiss_index_ids[I[0][i]]
                # return the index of the first element in faiss_index_ids that is equal to index_id
                start_id = self.faiss_index_ids.index(index_id)
                offset = I[0][i] - start_id
                top_k_entries.append({"url": self.knowledge_base[self.faiss_index_ids[I[0][i]]].url, "knowledge_base_index": self.faiss_index_ids[I[0][i]], "image_url": self.knowledge_base[self.faiss_index_ids[I[0][i]]].image_urls[offset],"similarity": D[0][i], "kb_entry": self.knowledge_base[self.faiss_index_ids[I[0][i]]]})
        return top_k_entries
    
    @torch.no_grad()
    def retrieve_image_faiss_batch(self, images, top_k=100, return_entry_list=False):
        """Retrieve the top K similar images from the knowledge base using faiss in batch.

        Args:
            images: The images to be compared.
            top_k (int): The number of top similar images to retrieve.
            pool_method (str): The pooling method to use ('max' or 'average').

        Returns:
            list: Top k entries, every entry is a dict of (url, kb_index, similarity)
        """ 
        # Process the input image
        if self.model_type == "clip":
            # CLIP Process the input image
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs.to(self.device)
            outputs = self.model(**inputs)
            image_scores = outputs.pooler_output
        elif self.model_type == "eva-clip":
            # EVA-CLIP Process the input image
            inputs = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device).half()
            image_scores = self.model.encode_image(inputs)
        assert self.faiss_index and self.faiss_index_ids is not None
        query = image_scores.float()
        query = torch.nn.functional.normalize(query, dim=-1)
        Ds, Is = self.faiss_index.search(query, top_k)
        top_k_list = []
        for D, I in zip(Ds, Is):
            top_k_entries = []
            for i in range(top_k):
                if return_entry_list:
                    top_k_entries.append(self.knowledge_base[self.faiss_index_ids[I[i]]])
                else:
                    top_k_entries.append({"url": self.knowledge_base[self.faiss_index_ids[I[i]]].url, "knowledge_base_index": self.faiss_index_ids[I[i]], "image_urls": self.knowledge_base[self.faiss_index_ids[I[i]]].image_urls,"similarity": D[i], "kb_entry": self.knowledge_base[self.faiss_index_ids[I[i]]]})
            top_k_list.append(top_k_entries)
            
        return top_k_list
            
    
    
            
        
        
        

