# from comet_ml import Experiment
import json
from argparse import ArgumentParser
from datetime import datetime
import os
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from torch.optim.lr_scheduler import OneCycleLR


# from data_utils import base_path, squarepad_transform, targetpad_transform, CIRRDataset, FashionIQDataset
from dataset import QFormerRerankerDataset, qformer_collate_fn

from data_utils import base_path, squarepad_transform, targetpad_transform
from torch.utils.data import RandomSampler, BatchSampler

# from utils import collate_fn, update_train_running_results,update_train_running_results_dict, set_train_bar_description_dict,set_train_bar_description, extract_index_blip_features, \
#     save_model, generate_randomized_fiq_caption, element_wise_sum, device

device = "cuda"


def update_train_running_results_dict(
    train_running_results: dict, loss_dict: dict, images_in_batch: int
):
    """
    Update `train_running_results` dict during training
    :param train_running_results: logging training dict
    :param loss: computed loss for batch
    :param images_in_batch: num images in the batch
    """
    for key in loss_dict.keys():
        if key not in train_running_results:
            train_running_results[key] = 0
        train_running_results[key] += (
            loss_dict[key].to("cpu", non_blocking=True).detach().item()
            * images_in_batch
        )

    train_running_results["images_in_epoch"] += images_in_batch


def set_train_bar_description_dict(
    train_bar, epoch: int, num_epochs: int, train_running_results: dict
):
    """
    Update tqdm train bar during training
    :param train_bar: tqdm training bar
    :param epoch: current epoch
    :param num_epochs: numbers of epochs
    :param train_running_results: logging training dict
    """
    images_in_epoch = train_running_results["images_in_epoch"]
    bar_content = ""
    for key in train_running_results:
        if key != "images_in_epoch":
            bar_content += (
                f"{key}: {train_running_results[key] / images_in_epoch:.3f}, "
            )
    train_bar.set_description(desc=f"[{epoch}/{num_epochs}] " f"{bar_content}")


def finetune_blip(
    num_epochs: int,
    blip_model_name: str,
    learning_rate: float,
    batch_size: int,
    validation_frequency: int,
    transform: str,
    save_training: bool,
    save_best: bool,
    checkpoint_path: str = None,
    blip2_checkpoint_path: str = None,
    **kwargs,
):
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/evqa_sim_{blip_model_name}_{training_start}"
    )
    training_path.mkdir(exist_ok=False, parents=True)

    with open(training_path / "training_hyperparameters.json", "w+") as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    blip_model, vis_processors, txt_processors = load_model_and_preprocess(
        name=blip_model_name, model_type="pretrain", is_eval=False, device="cuda"
    )
    update_method = getattr(blip_model, "_update_f_former", None)
    if callable(update_method):
        blip_model._update_f_former()

    input_dim = 224

    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print("Square pad preprocess pipeline is used")
    elif transform == "targetpad":
        target_ratio = kwargs["target_ratio"]
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f"Target pad with {target_ratio = } preprocess pipeline is used")
    else:
        raise ValueError(
            "Preprocess transform should be in ['clip', 'squarepad', 'targetpad']"
        )

    train_dataset = QFormerRerankerDataset(
        knowledge_base_file=kwargs["knowledge_base_file"],
        train_file=kwargs["train_file"],
        negative_db_file=kwargs["negative_db_file"],
        inat_id2name=kwargs["inat_id2name"],
        preprocess=preprocess,
        use_negative=True,
        neg_num=24,
    )
    print("Training datset length: ", len(train_dataset))
    if kwargs["eval_file"] is not None:
        eval_dataset = QFormerRerankerDataset(
            knowledge_base_file=kwargs["knowledge_base_file"],
            train_file=kwargs["eval_file"],
            inat_id2name=kwargs["inat_id2name"],
            preprocess=preprocess,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=qformer_collate_fn,
            num_workers=kwargs["num_workers"],
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
    train_dataloader = DataLoader(
        train_dataset,
        pin_memory=True,
        collate_fn=qformer_collate_fn,
        num_workers=kwargs["num_workers"],
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    print("Training dataloader length: ", len(train_dataloader))

    optimizer = optim.AdamW(
        [
            {
                "params": filter(lambda p: p.requires_grad, blip_model.parameters()),
                "lr": learning_rate,
                "betas": (0.9, 0.98),
                "eps": 1e-7,
                "weight_decay": 0.05,
            }
        ]
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        pct_start=1.5 / num_epochs,
        div_factor=100.0,
        steps_per_epoch=len(train_dataloader),
        epochs=num_epochs,
    )

    scaler = torch.cuda.amp.GradScaler()

    if checkpoint_path and blip2_checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        blip_model.load_state_dict(blip2_checkpoint_path)
        print(f"blip2 model loaded from {blip2_checkpoint_path}")
        print(f"Checkpoint loaded, resuming training from epoch {start_epoch}")

    training_log_frame = pd.DataFrame()
    # validation_log_frame = pd.DataFrame()
    blip_model.use_vanilla_qformer = True
    print("Training loop started")
    for epoch in range(num_epochs):
        train_running_results = {"images_in_epoch": 0}
        train_bar = tqdm(train_dataloader, ncols=150)
        blip_model.train()
        for idx, (
            reference_images,
            questions,
            positive_articles,
            negative_articles,
            _,
        ) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            step = len(train_bar) * epoch + idx
            optimizer.zero_grad()
            reference_images = reference_images.to(device, non_blocking=True)
            positive_articles = [
                txt_processors["eval"](positive_article)
                for positive_article in positive_articles
            ]
            negative_articles = [
                [txt_processors["eval"](article) for article in negative_article]
                for negative_article in negative_articles
            ]  # negative_articles is [batch_size, neg_num&2]
            questions = [txt_processors["eval"](question) for question in questions]
            # Extract the features, compute the logits and the loss
            with torch.cuda.amp.autocast():
                loss_dict = blip_model(
                    {
                        "image": reference_images,
                        "question": questions,
                        "article": positive_articles,
                        "negative": negative_articles,
                    }
                )
                loss = 0.0
                for key in loss_dict.keys():
                    loss += loss_dict[key]
            # Backpropagate and update the weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            update_train_running_results_dict(
                train_running_results, loss_dict, images_in_batch
            )
            set_train_bar_description_dict(
                train_bar, epoch, num_epochs, train_running_results
            )
            if save_training and (idx + 1) % kwargs["save_frequency"] == 0:
                checkpoint = {
                    "epoch": epoch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }
                torch.save(checkpoint, str(training_path / "checkpoint.pth"))
                torch.save(
                    blip_model.state_dict(),
                    str(
                        training_path / f"model_{epoch*len(train_dataloader) + idx}.pth"
                    ),
                )

        loss_log_dict = {"epoch": epoch}
        for key in train_running_results.keys():
            if key != "images_in_epoch":
                loss_log_dict[key] = float(
                    train_running_results[key]
                    / train_running_results["images_in_epoch"]
                )
            # Training CSV logging
        training_log_frame = pd.concat(
            [training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])]
        )
        training_log_frame.to_csv(str(training_path / "train_metrics.csv"), index=False)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument(
        "--num-epochs", default=300, type=int, help="number training epochs"
    )
    parser.add_argument(
        "--blip-model-name",
        default="blip2_cir_cat",
        type=str,
        help="[blip2_cir_cat, blip2_cir]",
    )
    parser.add_argument(
        "--learning-rate", default=2e-6, type=float, help="Learning rate"
    )
    parser.add_argument("--batch-size", default=512, type=int, help="Batch size")
    parser.add_argument(
        "--validation-frequency",
        default=1,
        type=int,
        help="Validation frequency expressed in epochs",
    )
    parser.add_argument(
        "--target-ratio", default=1.25, type=float, help="TargetPad target ratio"
    )
    parser.add_argument(
        "--transform",
        default="targetpad",
        type=str,
        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ",
    )
    parser.add_argument(
        "--save-training",
        dest="save_training",
        action="store_true",
        help="Whether save the training model",
    )
    parser.add_argument(
        "--save-best",
        dest="save_best",
        action="store_true",
        help="Save only the best model during training",
    )
    parser.add_argument(
        "--save_frequency",
        default=50000,
        type=int,
        help="Save frequency expressed in steps",
    )
    parser.add_argument(
        "--checkpoint-path", default=None, type=str, help="Path to the checkpoint"
    )
    parser.add_argument(
        "--blip2-checkpoint-path",
        default=None,
        type=str,
        help="Path to the blip2 checkpoint",
    )
    parser.add_argument("--train_file", type=str, help="Path to the training file")
    parser.add_argument(
        "--eval_file", default=None, type=str, help="Path to the evaluation file"
    )
    parser.add_argument(
        "--knowledge_base_file", type=str, help="Path to the knowledge base file"
    )
    parser.add_argument(
        "--negative_db_file", type=str, help="Path to the negative db file"
    )
    parser.add_argument(
        "--inat_id2name", type=str, help="Path to the inat_id2name.json"
    )

    args = parser.parse_args()

    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "num_workers": args.num_workers,
        "blip_model_name": args.blip_model_name,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "checkpoint_path": args.checkpoint_path,
        "blip2_checkpoint_path": args.blip2_checkpoint_path,
        "save_frequency": args.save_frequency,
        "data_path": args.data_path,
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "knowledge_base_file": args.knowledge_base_file,
        "negative_db_file": args.negative_db_file,
        "inat_id2name": args.inat_id2name,
    }

    finetune_blip(**training_hyper_params)
