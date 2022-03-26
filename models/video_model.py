from typing import Dict

import neptune.new as neptune
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torchvision.models.video import r2plus1d_18

from config import NEPTUNE_API_TOKEN, NEPTUNE_PROJECT_NAME
from models.model_loader import ModelLoader
from utils.summary_loss import SummaryLoss


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
            general_parameters: Dict = None,
            train_parameters: Dict = None,
            feature_extractor_parameters: Dict = None,
            transformer_parameters: Dict = None,
            heads: Dict = None,
            freeze_scheduler: Dict = None,
            loss_function=nn.BCEWithLogitsLoss,
            steps_per_epoch: int = 1000,
    ):
        """
        Args:
            general_parameters (Dict): Dict containing general parameters not parameterizing training process.
                [Warning] Must contain fields:
                    - path_to_save (str)
                    - neptune (bool)
            feature_extractor_parameters (Dict): Dict containing parameters regarding currently used feature extractor.
                [Warning] Must contain fields:
                    - "name" (str)
                    - "representation_size" (int)
            train_parameters (Dict): Dict containing parameters parameterizing the training process.
                [Warning] Must contain fields:
                    - "num_segments" (int)
                    - "lr" (float)
                    - "multiply_lr_step" (float)
                    - "warmup_steps" (float)
                    - "classification_mode" (str)
            heads (Dict): Dict containg information describing structure of output heads for specific tasks (gloss/hamnosys).
            freeze_scheduler (Dict): Dict containing information describing feature_extractor & transformer freezing/unfreezing process.
            loss_function (torch.nn.Module): Loss function.
        """
        super().__init__()

        if general_parameters["neptune"]:
            tags = [
                train_parameters["classification_mode"],
                "CustomVideoResNet"
            ]
            self.run = initialize_neptun(tags)
            self.run["parameters"] = {
                "general_parameters": general_parameters,
                "train_parameters": general_parameters,
                "feature_extractor_parameters": feature_extractor_parameters,
                "heads": heads,
                "freeze_scheduler": freeze_scheduler,
                "loss_function": loss_function,
            }
        else:
            self.run = None

        # parameters
        self.lr = train_parameters["lr"]
        self.model_save_dir = general_parameters["path_to_save"]
        self.warmup_steps = train_parameters["warmup_steps"]
        self.multiply_lr_step = train_parameters["multiply_lr_step"]
        self.classification_heads = heads[train_parameters["classification_mode"]]
        self.num_segments = train_parameters["num_segments"]
        self.num_channels = 3

        # heads
        self.cls_head = []
        self.loss_weights = []
        for value in self.classification_heads.values():
            self.cls_head.append(
                nn.Linear(feature_extractor_parameters["representation_size"], value["num_class"])
            )
            self.loss_weights.append(value["loss_weight"])

        # losses
        self.summary_loss = SummaryLoss(loss_function, self.loss_weights)

        # models-parts
        self.model_loader = ModelLoader()
        self.multi_frame_feature_extractor = r2plus1d_18(True)

    def forward(self, input, **kwargs):
        predictions = []
        input = input.to(self.device).reshape(-1, self.num_channels, self.num_segments, 224, 224)
        x = self.multi_frame_feature_extractor(input)
        for head in self.cls_head:
            predictions.append(head(x.cpu()))
        return predictions

    def training_step(self, batch, batch_idx):
        input, _, targets = batch
        predictions = self(input)
        loss = self.summary_loss(predictions, targets)
        if self.run:
            self.run["metrics/batch/training_loss"].log(loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input, _, targets = batch
        predictions = self(input)
        loss = self.summary_loss(predictions, targets)
        if self.run:
            self.run["metrics/batch/validation_loss"].log(loss)
        return {"val_loss": loss, "targets": targets, "predictions": predictions}

    def validation_epoch_end(self, out):
        head_names = list(self.classification_heads.keys())
        # initialize empty list with list per head
        all_targets = [[] for name in head_names]
        all_predictions = [[] for name in head_names]
        for single_batch in out:
            targets, predictions = single_batch["targets"], single_batch["predictions"]
            # append predictions and targets for every head
            for nr_head, head_targets in enumerate(targets):
                all_targets[nr_head] += list(torch.argmax(targets[nr_head], dim=1).cpu().detach().numpy())
                all_predictions[nr_head] += list(torch.argmax(predictions[nr_head], dim=1).cpu().detach().numpy())

        for nr_head, targets_for_head in enumerate(all_targets):
            head_name = head_names[nr_head]
            predictions_for_head = all_predictions[nr_head]
            head_report = "\n".join(
                [
                    head_name,
                    classification_report(
                        targets_for_head, predictions_for_head, zero_division=0
                    ),
                ]
            )
            print(head_report)
            f1 = f1_score(targets_for_head, predictions_for_head,
                          average='macro', zero_division=0)
            if self.run:
                log_path = "/".join(["metrics/epoch/", head_name])
                self.run[log_path].log(head_report)
                self.run[f'/metrics/epoch/f1/{head_name}'].log(f1)

        if self.trainer.global_step > 0:
            print("Saving model...")
            torch.save(self.state_dict(), self.model_save_dir)

    def configure_optimizers(self):
        # set optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.lr, max_lr=0.001, step_size_up=5,
                                                           mode="exp_range", gamma=0.85, cycle_momentum=False)

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
