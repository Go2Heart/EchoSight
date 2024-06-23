"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures

@registry.register_model("blip2_reranker")
class Blip2QformerReranker(Blip2Base):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }
    
    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=512,
        use_vanilla_qformer=False,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = 512 #TODO read from config
        self.use_vanilla_qformer = use_vanilla_qformer
        # new tokens
        self.prompt_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.Qformer.config.hidden_size)
        )
        self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)
    
    def forward(self, samples):
        image = samples["image"]
        question = samples["question"]
        article = samples["article"]
        negatives = samples["negative"]
        # image feature
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        
        # query tokens
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        
        # question tokens
        question_tokens = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        # fusion question and image tokens into a set of multi-modal tokens
        # print(query_atts.shape, question_tokens.attention_mask.shape)
        
        
        attention_mask = torch.cat([query_atts, question_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            question_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        if self.use_vanilla_qformer:
            question_output = fusion_output
        else:
            question_output = self.Qformer.bert(
                question_tokens.input_ids,
                query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
                attention_mask=attention_mask,
                return_dict=True,
            )
        
        fusion_feats = F.normalize(
            self.vision_proj(question_output.last_hidden_state[:, : query_tokens.size(1), :]), dim=-1
        )
        
        # fusion_feats = F.normalize(
        #     self.text_proj(question_output.last_hidden_state[:, 32, :]), dim=-1
        # )
        
        # article feature
        
        article_tokens = self.tokenizer(
            article,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        article_output = self.Qformer.bert(
            article_tokens.input_ids,
            attention_mask=article_tokens.attention_mask,
            return_dict=True,
        )
        # article_feats = F.normalize(
        #     self.text_proj(article_output.last_hidden_state[:, 0, :]), dim=-1 # TODO validate projection here
        # )
        
        ### ================== Negative ================== ###
        negatives = list(zip(*negatives)) # transpose the list to [batch_size, neg_num]
        negatives = [article for negative in negatives for article in negative] # flatten the list
        negative_tokens = self.tokenizer(
            negatives,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device) # [batch_size * neg_num, max_txt_len]
        negative_output = self.Qformer.bert(
            negative_tokens.input_ids,
            attention_mask=negative_tokens.attention_mask,
            return_dict=True,
        ) # [batch_size * neg_num, max_txt_len, hidden_size]
        negative_cls = negative_output.last_hidden_state[:, 0, :].view(image.size(0), -1, negative_output.last_hidden_state.size(-1)) # [batch_size, neg_num, hidden_size]
        # concat the positive with negative
        positive_cls = article_output.last_hidden_state[:, 0, :].unsqueeze(1)
        
        concatenated_cls = torch.cat([positive_cls, negative_cls], dim=1) # [batch_size, 1 + neg_num, hidden_size]
        
        concatenated_feats = F.normalize(
            self.text_proj(concatenated_cls), dim=-1
        ) # [batch_size, 1 + neg_num, embed_dim]
        
        sim_a2f = torch.matmul(
            concatenated_feats, fusion_feats.permute(0, 2, 1)
        )
        
        sim_q2a, _ = sim_a2f.max(-1)
        sim_q2a = sim_q2a / self.temp
        bs = image.size(0)

        targets = torch.linspace(0,  0, bs, dtype=int).to(
            image.device
        )
        
        loss_q2a = F.cross_entropy(sim_q2a, targets)
        
        return {
            'loss': loss_q2a
        }
    @torch.no_grad()
    def extract_visual_features(self, image):
        # image feature
        image_embeds = self.ln_vision(self.visual_encoder(image))
        return image_embeds
    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding='max_length', truncation=True).to(
                self.device
            )
            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state 
            text_features = self.text_proj(text_embeds) 
            text_features = F.normalize(text_features, dim=-1) 

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            fusion_output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            # multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]
            if self.use_vanilla_qformer:
                question_output = fusion_output
            else:
                question_output = self.Qformer.bert(
                    text.input_ids,
                    query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            multimodal_embeds = F.normalize(
                self.vision_proj(question_output.last_hidden_state[:, : query_tokens.size(1), :]), dim=-1
                # self.text_proj(question_output.last_hidden_state[:, 32, :]), dim=-1
            )

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model   
        