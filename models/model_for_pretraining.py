import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from config import NEPTUNE_API_TOKEN, NEPTUNE_PROJECT_NAME
import neptune.new as neptune
from torch.optim.lr_scheduler import MultiplicativeLR
from sklearn.metrics import classification_report
from utils.classification_mode import create_heads_dict
from utils.summary_loss import SummaryLoss

from models.feature_extractors.multi_frame_feature_extractor import (
    MultiFrameFeatureExtractor,
)
from models.model_loader import ModelLoader
# initialize neptune logging
def initialize_neptun(tags):
    return neptune.init(
        api_token=NEPTUNE_API_TOKEN,
        project=NEPTUNE_PROJECT_NAME,
        tags=tags,
        capture_stdout=False,
        capture_stderr=False,
    )


class PreTrainingModel(pl.LightningModule):
    """Awesome model for Gloss Translation"""

    def __init__(
        self,
        model_config = {
        "lr": 1e-5,
        "multiply_lr_step": 0.7,
        "warmup_steps": 100.0,
        "transformer_output_size": 1024,
        "representation_size": 2048,
        "feedforward_size": 4096,
        "num_encoder_layers": 1,
        "num_segments": 8,
        "num_attention_heads": 16,
        "classification_mode": "gloss",
        "feature_extractor_name": "cnn_extractor",
        "feature_extractor_model_path": "efficientnet_b1",
        "transformer_name": "fake_transformer",
        "model_save_dir": "",
        "neptune": False,}
    ):
        super().__init__()

        if model_config["neptune"]:
            tags = [model_config["classification_mode"],
            model_config["feature_extractor_name"],
            model_config["transformer_name"],
            "pre-training"]
            self.run = initialize_neptun(tags)
        else:
            self.run = None
 
        # parameters
        self.lr = model_config["lr"]
        self.model_save_dir = model_config["model_save_dir"]
        self.warmup_steps = model_config["warmup_steps"]
        self.multiply_lr_step = model_config["multiply_lr_step"]
        self.num_classes_dict = create_heads_dict(model_config["classification_mode"])
        self.cls_head = []
        self.loss_weights = []
        for value in self.num_classes_dict.values():
            self.cls_head.append(nn.Linear(model_config["representation_size"], value["num_class"]))
            self.loss_weights.append(value["loss_weight"])

        # losses
        self.summary_loss = SummaryLoss(nn.CrossEntropyLoss, self.loss_weights)

        # models-parts
        self.model_loader = ModelLoader()
        self.feature_extractor = self.model_loader.load_feature_extractor(
            model_config["feature_extractor_name"],
            model_config["representation_size"],
            model_path=model_config["feature_extractor_model_path"],
        )
        self.multi_frame_feature_extractor = MultiFrameFeatureExtractor(
            self.feature_extractor
        )

    def forward(self, input, **kwargs):
        predictions = []
        x = self.multi_frame_feature_extractor(input.to(self.device))
        for head in self.cls_head:
            predictions.append(head(x.cpu()))
        return predictions

    def training_step(self, batch, batch_idx):
        input, target = batch
        targets = target["target"]
        predictions = self(input)
        loss = self.summary_loss(predictions, targets)
        if self.run:
            self.run["metrics/batch/training_loss"].log(loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input, target = batch
        targets = target["target"]
        predictions = self(input)
        loss = self.summary_loss(predictions, targets)
        if self.run:
            self.run["metrics/batch/validation_loss"].log(loss)
        return {"loss": loss, "targets": targets, "predictions": predictions}

    def validation_epoch_end(self, out):
        head_names = list(self.num_classes_dict.keys())
        # initialize empty list with list per head
        all_targets = [[] for name in head_names]
        all_predictions = [[] for name in head_names]
        for single_batch in out:
            targets, predictions = single_batch["targets"], single_batch["predictions"]
            # append predictions and targets for every head
            for nr_head, head_targets in enumerate(targets):
                all_targets[nr_head].append(
                    torch.argmax(targets[nr_head]).cpu().detach().numpy()
                )
                all_predictions[nr_head].append(
                    torch.argmax(predictions[nr_head]).cpu().detach().numpy()
                )

        for nr_head, head_targets in enumerate(all_targets):
            head_report = "\n".join(
                [
                    head_names[nr_head],
                    classification_report(
                        all_targets[nr_head], all_predictions[nr_head], zero_division=0
                    ),
                ]
            )
            print(head_report)
            if self.run:
                log_path = "/".join(["metrics/epoch/", head_names[nr_head]])
                self.run[log_path].log(head_report)

        if self.trainer.global_step > 0:
            print("Saving model...")
            torch.save(self.state_dict(), self.model_save_dir)

    def configure_optimizers(self):
        # set optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # set scheduler: multiply lr every epoch
        def lambd(epoch):
            return self.multiply_lr_step

        scheduler = MultiplicativeLR(optimizer, lr_lambda=lambd)
        return [optimizer], [scheduler]

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # set warm-up
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        optimizer.step(closure=optimizer_closure)
        if self.run:
            self.run["params/lr"].log(self.lr)
