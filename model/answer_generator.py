""" Serve for question generation for a list of given knowledge base entries. 
"""

import re
from typing import List
from openai import OpenAI
import torch
from .retriever import WikipediaKnowledgeBaseEntry
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers.generation import GenerationConfig
# from openai import OpenAI
import time


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def reconstruct_wiki_article(knowledge_entry: WikipediaKnowledgeBaseEntry):
    title = knowledge_entry.title
    article = "# Wiki Article: " + title + "\n"
    for it, section_title in enumerate(knowledge_entry.section_titles):
        if "external link" in section_title.lower() or "reference" in section_title.lower():
            continue
        article += "\n## Section Title: " + section_title + "\n" + knowledge_entry.section_texts[it]
    
    return article

def reconstruct_wiki_sections(knowledge_entry, section_index=-1):
    title = knowledge_entry.title
    sections = []
    for it, section_title in enumerate(knowledge_entry.section_titles):
        if it == int(section_index):
            evidence_section = "# Wiki Article: " + title + "\n" + "## Section Title: " + section_title + "\n" + knowledge_entry.section_texts[it]
        elif "external links" in section_title.lower() or "references" in section_title.lower():
            continue
        else:
            sections.append("# Wiki Article: " + title + "\n" + "## Section Title: " + section_title + "\n" + knowledge_entry.section_texts[it])
    if section_index != -1:
        return evidence_section, sections
    return sections

def get_all_sections(knowledge_entry):
    sections = []
    for it, section_title in enumerate(knowledge_entry.section_titles):
        sections.append("* Section Title: " + section_title + "\n" + knowledge_entry.section_texts[it])
    
    return sections

pseudo_tokenizer = None

def _adjust_prompt_length(prompt, desired_token_length):
    global pseudo_tokenizer
    
    if pseudo_tokenizer is None:
        pseudo_tokenizer = AutoTokenizer.from_pretrained("/remote-home/share/huggingface_model/Mistral-7B-Instruct-v0.2")

    # Tokenize the prompt
    tokens = pseudo_tokenizer.encode(prompt)

    if len(tokens) > desired_token_length:
        # If the prompt is too long, trim it
        trimmed_tokens = tokens[:desired_token_length]
        # Convert tokens back to text
        trimmed_text = pseudo_tokenizer.decode(trimmed_tokens, clean_up_tokenization_spaces=True)[4:]
        return trimmed_text
    else:
        # If the length is already as desired
        return prompt
class AnswerGenerator:
    """ Question generator for EchoSight.

    """
    def __init__(self):
        self.model = None
        
    def load_model(self, model_name):
        """Load the model.

        Args:
            model_name: The model to load.
        """
        raise NotImplementedError
    
    
    

class MistralAnswerGenerator(AnswerGenerator):
    """ Mistral Question generator for EchoSight.
    """
    def __init__(self, device, model_path, use_embedding_model=False):
        """Initialize the QuestionGenerator class.
        """
        super().__init__()
        self.device = device
        self.model_path = model_path
        self._load_model()
        if use_embedding_model:
            self._load_embedding()
        else:
            self.emb = None
        
    def _load_model(self):
        """Load the model.
        """
        disable_torch_init()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16).eval()
        self.model.to(self.device)
    
    @torch.no_grad()
    def llm_answering(self, question, entry=None, entry_dict=None, entry_section=None, oracle_setting="subject", evidence_sec=None):
        """Answer the question for a given entry"""
        if entry is not None:
            context = reconstruct_wiki_article(entry)
            context = _adjust_prompt_length(context, 4096)
            prompt = "Context: " + context + "\nQuestion: " + question + "\nThe answer is:"
            
        elif entry_dict is not None:
            context = reconstruct_wiki_article(WikipediaKnowledgeBaseEntry(entry_dict))
            context = _adjust_prompt_length(context, 4096)
            prompt = "Context: " + context + "\nQuestion: " + question + "\nThe answer is:"
            # prompt = "Entity name: " + WikipediaKnowledgeBaseEntry(entry_dict).title + "\nQuestion: " + question + "\nThe answer is:"#
        elif entry_section is not None:
            prompt = "Context: " + entry_section + "\nQuestion: " + question + "\nThe answer is:"
        else:
            prompt = "Question: " + question + "\nThe answer is:"
        
        messages = [
                {"role": "user", "content": prompt},
        ]
        
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt", max_length=8000, truncation=True)
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=128, do_sample=True, top_p=0.9, temperature=0.9,  pad_token_id=2)
        response = (self.tokenizer.decode(generated_ids[0][model_inputs.shape[1]:]))[:-4]
        
        return response
class LLaMA3AnswerGenerator(AnswerGenerator):
    def __init__(self, device, model_path):
        """Initialize the QuestionGenerator class.
        """
        super().__init__()
        self.device = device
        self.model_path = model_path
        self._load_model()
        
    def _load_model(self):
        """Load the model.

        Args:
            model_path: The model to load.
        """
        disable_torch_init()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16).eval()
        self.model.to(self.device)

    @torch.no_grad()
    def llm_answering(self, question, entry=None, entry_dict=None, entry_section=None, oracle_setting="subject", evidence_sec=None):
        """Answer the question for a given entry"""
        if entry is not None:
            context = reconstruct_wiki_article(entry)
            context = _adjust_prompt_length(context, 4096)
            prompt = "Context: " + context + "\nQuestion: " + question + "\nThe answer is:"
            
        elif entry_dict is not None:
            context = reconstruct_wiki_article(WikipediaKnowledgeBaseEntry(entry_dict))
            prompt = "Entity name: " + WikipediaKnowledgeBaseEntry(entry_dict).title + "\nQuestion: " + question + "\nThe answer is:"
        elif entry_section is not None:
            prompt = "Context: " + entry_section + "\nQuestion: " + question + "\nThe answer is:"
        else:
            # vanilla setting
            prompt = "Question: " + question + "\nThe answer is:"
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant for answering encyclopedic questions."},
            {"role": "user", "content": prompt},
        ]
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt", max_length=8000, truncation=True)
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(
            model_inputs,
            max_new_tokens=128,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = (self.tokenizer.decode(generated_ids[0][model_inputs.shape[1]:]))
        
        return response
        
    

class GPT4AnswerGenerator(AnswerGenerator):
    """ OpenAI Question generator for EchoSight.

    """
    def __init__(self):
        """ Initialize the QuestionGenerator class.
        """
        super().__init__()
        self.client = OpenAI(
            api_key="YOUR_API_KEY"
            )
        

    
    def get_gpt4_answer(self, prompt_str):
        MAX_RETRIES = 5
        retries = 0

        while retries < MAX_RETRIES:
            try:
                completion = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_str}
                ]
                )

                assistant_reply = (completion.choices[0].message.content)
                break  
            except Exception as e:
                print(f"Error: {str(e)}")
                retries += 1
                time.sleep(2)
        return assistant_reply
    def llm_answering(self, question, entry=None, entry_dict=None, entry_section=None):
        """Answer the question for a given entry"""
        if entry is not None:
            context = reconstruct_wiki_article(entry)
            context = _adjust_prompt_length(context, 4096)
            
            prompt = "Context: " + context + "\nQuestion: " + question + "\nThe answer is:"
        elif entry_dict is not None:
            context = reconstruct_wiki_article(WikipediaKnowledgeBaseEntry(entry_dict))
            context = _adjust_prompt_length(context, 4096)
            prompt = "Context: " + context + "\nQuestion: " + question + "\nThe answer is:"
        elif entry_section is not None:
            prompt = "Context: " + entry_section + "\nQuestion: " + question + "\nThe answer is:"
        else:
            prompt = "Question: " + question + "\nThe answer is:"
        response = self.get_gpt4_answer(prompt)
        return response
    
    
class PaLMAnswerGenerator(AnswerGenerator):
    """ Google PaLM Question generator for EchoSight.

    """
    def __init__(self):
        """ Initialize the QuestionGenerator class.
        """
        super().__init__()
        import vertexai
        from vertexai.preview.language_models import (ChatModel, InputOutputTextPair,
                                              TextEmbeddingModel,
                                              TextGenerationModel)
        import os
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="YOUR_CREDENTIALS.json"
        PROJECT_ID = "YOUR_PROJECT_ID"
        REGION = "YOUR_REGION"
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = TextGenerationModel.from_pretrained("text-bison@002")
        
    def llm_answering(self, question, entry=None, entry_dict=None, entry_section=None):
        """Answer the question for a given entry"""
        if entry is not None:
            context = reconstruct_wiki_article(entry)
            context = _adjust_prompt_length(context, 4096)
            
            prompt = "Context: " + context + "\nQuestion: " + question + "\nThe answer is:"
        elif entry_dict is not None:
            context = reconstruct_wiki_article(WikipediaKnowledgeBaseEntry(entry_dict))
            context = _adjust_prompt_length(context, 4096)
            prompt = "Context: " + context + "\nQuestion: " + question + "\nThe answer is:"
        elif entry_section is not None:
            prompt = "Context: " + entry_section + "\nQuestion: " + question + "\nThe answer is:"
        else:
            prompt = "Question: " + question + "\nThe answer is:"
        
        response = self.model.predict(prompt, 
                                    temperature=0.2,
                                    max_output_tokens=128,
                                    top_k=40,
                                    top_p=0.95,
        ).text
        return response
    
    


class BgeTextReranker:
    def __init__(self, model_path, device):
        """Initialize the Text Reranker
        """
        self.device = device
        self.model_path = model_path
        self._load_model()
    def _load_model(self):
        """Load the model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
    @torch.no_grad()
    def rerank_entry_sections(self, question, sections, top_k=3, gt_index=-1):
        if gt_index == -1:
            return -1, 0 
        pairs = [[question, section] for section in sections[:top_k]]
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=6000).to(self.device)
        scores = self.model(**inputs, return_dict=True).logits.view(-1,).float()
        scores, index = torch.sort(scores, descending=True)
        
        return index[0], int(index[0])==int(gt_index)
  


        
        
        
        
        
        
        
        