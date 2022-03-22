from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from config import NEPTUNE_API_TOKEN, NEPTUNE_PROJECT_NAME
from models.feature_extractors.multi_frame_feature_extractor import (
    MultiFrameFeatureExtractor,
)
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


class PreTrainingModel(pl.LightningModule):
    """Awesome model for Gloss Translation"""

    def __init__(
        self,
        general_parameters: Dict = None,
        train_parameters: Dict = None,
        feature_extractor_parameters: Dict = None,
        transformer_parameters: Dict = None,
        heads: Dict = None,
        freeze_scheduler: Dict = None,
        steps_per_epoch: int = 1000
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
                    - "model_path" (str)
                    - "representation_size" (int)
            transformer_parameters (Dict): Dict containing parameters regarding currently used transformer.
                [Warning] Must contain fields:
                    - "name" (str)
                    - "output_size" (int)
                    - "feedforward_size" (int)
                    - "num_encoder_layers" (int)
                    - "num_attention_heads" (int)
                    - "dropout_rate" (float)
            train_parameters (Dict): Dict containing parameters parameterizing the training process.
                [Warning] Must contain fields:
                    - "num_segments" (int)
                    - "lr" (float)
                    - "multiply_lr_step" (float)
                    - "warmup_steps" (float)
                    - "classification_mode" (str)
            heads (Dict): Dict containg information describing structure of output heads for specific tasks (gloss/hamnosys).
            freeze_scheduler (Dict): Dict containing information describing feature_extractor & transformer freezing/unfreezing process.
        """
        super().__init__()
        if general_parameters["neptune"]:
            tags = [train_parameters["classification_mode"], feature_extractor_parameters["name"], transformer_parameters["name"], "pre_training"]
            self.run = initialize_neptun(tags)
            self.run["parameters"] = {
                "general_parameters": general_parameters,
                "train_parameters": general_parameters,
                "feature_extractor_parameters": feature_extractor_parameters,
                "transformer_parameters": transformer_parameters,
                "heads": heads,
                "freeze_scheduler": freeze_scheduler
            }
        else:
            self.run = None

        # parameters
        self.lr = train_parameters["lr"]
        self.model_save_dir = general_parameters["path_to_save"]
        self.warmup_steps = train_parameters["warmup_steps"]
        self.multiply_lr_step = train_parameters["multiply_lr_step"]
        self.classification_heads = heads[train_parameters['classification_mode']]
        self.cls_head = []
        self.loss_weights = []
        for value in self.classification_heads.values():
            self.cls_head.append(nn.Linear(feature_extractor_parameters["representation_size"], value["num_class"]))
            self.loss_weights.append(value["loss_weight"])

        # losses
        self.summary_loss = SummaryLoss(nn.CrossEntropyLoss, self.loss_weights)

        # models-parts
        self.model_loader = ModelLoader()
        self.feature_extractor = self.model_loader.load_feature_extractor(
                feature_extractor_name=feature_extractor_parameters["name"],
                representation_size=feature_extractor_parameters["representation_size"],
                model_path=feature_extractor_parameters["model_path"],
            )
        self.multi_frame_feature_extractor = MultiFrameFeatureExtractor(
            self.feature_extractor
        )

        self.steps_per_epoch = steps_per_epoch

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
        head_names = list(self.classification_heads.keys())
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
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr)

        steps_per_epoch = self.steps_per_epoch // self.trainer.accumulate_grad_batches
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, epochs=self.trainer.max_epochs,
                                                             steps_per_epoch=steps_per_epoch)
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

        optimizer.step(closure=optimizer_closure)
        if self.run:
            self.run["params/lr"].log(self.lr)
