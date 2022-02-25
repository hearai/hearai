import neptune.new as neptune
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from config import NEPTUNE_API_TOKEN, NEPTUNE_PROJECT_NAME
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import MultiplicativeLR
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


class GlossTranslationModel(pl.LightningModule):
    """Awesome model for Gloss Translation"""

    def __init__(
        self,
        MODEL_CONFIG = {"lr": 1e-5,
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
        "neptune": False,
        "device": "cpu",}
    ):
        super().__init__()

        if neptune:
            tags = [MODEL_CONFIG["classification_mode"], MODEL_CONFIG["feature_extractor_name"], MODEL_CONFIG["transformer_name"]]
            self.run = initialize_neptun(tags)
        else:
            self.run = None

        # parameters
        self.lr = MODEL_CONFIG["lr"]
        self.model_save_dir = MODEL_CONFIG["model_save_dir"]
        self.warmup_steps = MODEL_CONFIG["warmup_steps"]
        self.multiply_lr_step = MODEL_CONFIG["multiply_lr_step"]
        self.num_classes_dict = create_heads_dict(MODEL_CONFIG["classification_mode"])

        # losses
        self.summary_loss = SummaryLoss(nn.CrossEntropyLoss)

        # models-parts
        self.model_loader = ModelLoader()
        self.feature_extractor = self.model_loader.load_feature_extractor(
            feature_extractor_name,
            representation_size,
            device=device,
            model_path=feature_extractor_model_path,
        )
        self.multi_frame_feature_extractor = MultiFrameFeatureExtractor(
            self.feature_extractor
        )
        if transformer_name == "sign_language_transformer":
            self.transformer = self.model_loader.load_transformer(
                transformer_name,
                representation_size,
                transformer_output_size,
                feedforward_size,
                num_encoder_layers,
                num_segments,
                num_attention_heads,
                device=device
            )
        else:
            self.transformer = self.model_loader.load_transformer(
                transformer_name, representation_size, transformer_output_size
            )
        self.cls_head = []
        print(self.num_classes_dict)
        for value in self.num_classes_dict.values():
            self.cls_head.append(nn.Linear(transformer_output_size, value))

    def forward(self, input, **kwargs):
        predictions = []
        x = self.multi_frame_feature_extractor(input)
        x = self.transformer(x)
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
        return {"val_loss": loss, "targets": targets, "predictions": predictions}

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
            self.scheduler.step()

    def configure_optimizers(self):
        # set optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # set scheduler: multiply lr every epoch
        def lambd(epoch):
            return self.multiply_lr_step

        self.scheduler = MultiplicativeLR(optimizer, lr_lambda=lambd)
        return [optimizer], [self.scheduler]

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
            self.run["params/lr"].log(optimizer.param_groups[0]["lr"])
