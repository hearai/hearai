import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from config import NEPTUNE_API_TOKEN, NEPTUNE_PROJECT_NAME
import neptune.new as neptune
from torch.optim.lr_scheduler import MultiplicativeLR
from sklearn.metrics import classification_report
from models.model_loader import ModelLoader
from models.feature_extractors.multi_frame_feature_extractor import (
    MultiFrameFeatureExtractor,
)
from utils.classification_mode import create_heads_dict


# initialize neptune logging
def initialize_neptun():
    return neptune.init(
        api_token=NEPTUNE_API_TOKEN,
        project=NEPTUNE_PROJECT_NAME
        )


def summary_loss(predictions, targets):
    loss = nn.CrossEntropyLoss()
    losses = []
    loss_sum = 0
    for prediction, target in zip(predictions, targets):
        one_loss = loss(prediction.to("cpu"), target.to("cpu"))
        losses.append(one_loss)
        loss_sum = loss_sum + one_loss
    return loss_sum


class GlossTranslationModel(pl.LightningModule):
    """Awesome model for Gloss Translation"""

    def __init__(
        self,
        lr=1e-5,
        multiply_lr_step=0.7,
        warmup_steps=100.0,
        transformer_output_size=1024,
        representation_size=2048,
        num_segments=10,
        classification_mode="gloss",
        feature_extractor_name="cnn_extractor",
        transformer_name="vanilla_transformer",
        model_save_dir="",
        neptune=False,
    ):
        super().__init__()

        if neptune:
            self.run = initialize_neptun()
        else:
            self.run = None

        # parameters
        self.lr = lr
        self.model_save_dir = model_save_dir
        self.warmup_steps = warmup_steps
        self.multiply_lr_step = multiply_lr_step
        self.num_classes_dict = create_heads_dict(classification_mode)

        # losses
        # self.summary_loss = summary_loss(predictions, targets)

        # models-parts
        self.model_loader = ModelLoader()
        self.feature_extractor = self.model_loader.load_feature_extractor(
            feature_extractor_name, representation_size
        )
        self.multi_frame_feature_extractor = MultiFrameFeatureExtractor(
            self.feature_extractor
        )
        self.transformer = self.model_loader.load_transformer(
            transformer_name, representation_size, transformer_output_size, num_segments
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
            predictions.append(head(x))
        return predictions

    def training_step(self, batch, batch_idx):
        input, targets = batch
        predictions = self(input)
        loss = summary_loss(predictions, targets)
        if self.run:
            self.run["metrics/batch/training_loss"].log(loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input, targets = batch
        predictions = self(input)
        loss = summary_loss(predictions, targets)
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
