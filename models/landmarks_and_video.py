from typing import Dict

import neptune.new as neptune
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from config import NEPTUNE_API_TOKEN, NEPTUNE_PROJECT_NAME
from sklearn.metrics import classification_report, f1_score
from utils.summary_loss import SummaryLoss
from torchvision import models

from models.feature_extractors.multi_frame_feature_extractor import (
    MultiFrameFeatureExtractor,
)
from models.model_loader import ModelLoader
from models.landmarks_models.lanmdarks_sequential_model import LandmarksSequentialModel
from models.landmarks_models.features_sequential_model import FeaturesSequentialModel
from models.head_models.head_sequential_model import HeadClassificationSequentialModel

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
            loss_function (torch.nn.Module): Loss function.
        """
        super().__init__()

        if general_parameters["neptune"]:
            tags = [
                train_parameters["classification_mode"],
                "pretrained_r2plus1d_18",
                transformer_parameters["name"],
                "video_and_landmarks",
            ]
            self.run = initialize_neptun(tags)
            self.run["parameters"] = {
                "general_parameters": general_parameters,
                "train_parameters": train_parameters,
                "feature_extractor_parameters": feature_extractor_parameters,
                "transformer_parameters": transformer_parameters,
                "heads": heads,
                "loss_function": loss_function,
            }
        else:
            self.run = None

        # parameters
        self.num_segments = train_parameters["num_segments"]
        self.num_channels = 3
        self.lr = train_parameters["lr"]
        self.model_save_dir = general_parameters["path_to_save"]
        self.warmup_steps = train_parameters["warmup_steps"]
        self.multiply_lr_step = train_parameters["multiply_lr_step"]
        self.use_frames = train_parameters["use_frames"]
        self.use_landmarks = train_parameters["use_landmarks"]
        self.classification_heads = heads[train_parameters["classification_mode"]]
        self.cls_head = nn.ModuleList()
        self.loss_weights = []
        for value in self.classification_heads.values():
            self.cls_head.append(
                HeadClassificationSequentialModel(
                    classes_number=value["num_class"],
                    representation_size=transformer_parameters["output_size"]
                    + feature_extractor_parameters["representation_size"],
                    additional_layers=heads["model"]["additional_layers"],
                    dropout_rate=heads["model"]["dropout_rate"],
                )
            )
            self.loss_weights.append(value["loss_weight"])

        # losses
        self.summary_loss = SummaryLoss(loss_function, self.loss_weights)

        # models-parts
        self.model_loader = ModelLoader()

        self.multi_frame_feature_extractor = models.video.r2plus1d_18(False)
        if feature_extractor_parameters["load_pretrained"]:
            self.multi_frame_feature_extractor.load_state_dict(
                torch.load(
                    "/dih4/dih4_2/hearai/amikolajczyk/hearai/pretrained_r2plus1d_18.ckpt"
                )
            )

        self.landmarks_model = LandmarksSequentialModel(
            feature_extractor_parameters["representation_size"],
            transformer_parameters["dropout_rate"],
        )
        self.pretransformer_model = FeaturesSequentialModel(
            feature_extractor_parameters["representation_size"],
            transformer_parameters["dropout_rate"],
        )

        self.transformer = self.model_loader.load_transformer(
            transformer_name=transformer_parameters["name"],
            feature_extractor_parameters=feature_extractor_parameters,
            transformer_parameters=transformer_parameters,
            train_parameters=train_parameters,
        )

        self.steps_per_epoch = steps_per_epoch

    def forward(self, input, **kwargs):
        predictions = []
        frames, landmarks = input

        im_width = frames.shape[-1]
        frames = frames.to(self.device).reshape(
            -1, self.num_channels, self.num_segments, im_width, im_width
        )
        x_frames = self.multi_frame_feature_extractor(frames.to(self.device))
        x_landmarks = self._prepare_landmarks_tensor(landmarks)
        x_landmarks = self.landmarks_model(x_landmarks)
        x = self.pretransformer_model(x_landmarks)

        x = self.transformer(x)
        x = torch.concat([x_frames, x], dim=-1)
        for head in self.cls_head:
            predictions.append(head(x))
        return predictions

    def _prepare_landmarks_tensor(self, landmarks):
        concatenated_landmarks = np.concatenate(
            [landmarks[landmarks_name] for landmarks_name in landmarks.keys()], axis=-1
        )
        return torch.as_tensor(
            concatenated_landmarks, dtype=torch.float32, device=self.device
        )

    def training_step(self, batch, batch_idx):
        targets, predictions, losses = self._process_batch(batch)
        self.scheduler.step()

        if self.run:
            self.run["metrics/batch/training_loss"].log(losses)
        return {"loss": losses}

    def validation_step(self, batch, batch_idx):
        targets, predictions, losses = self._process_batch(batch)
        if self.run:
            self.run["metrics/batch/validation_loss"].log(losses)
        return {"val_loss": losses, "targets": targets, "predictions": predictions}

    def _process_batch(self, batch):
        frames, landmarks, targets = batch
        predictions = self((frames, landmarks))
        losses = self.summary_loss(predictions, targets)
        return targets, predictions, losses

    def validation_epoch_end(self, out):
        head_names = list(self.classification_heads.keys())
        # initialize empty list with list per head
        all_targets = [[] for name in head_names]
        all_predictions = [[] for name in head_names]
        for single_batch in out:
            targets, predictions = single_batch["targets"], single_batch["predictions"]
            # append predictions and targets for every head
            for nr_head, head_targets in enumerate(targets):
                all_targets[nr_head] += list(
                    torch.argmax(targets[nr_head], dim=1).cpu().detach().numpy()
                )
                all_predictions[nr_head] += list(
                    torch.argmax(predictions[nr_head], dim=1).cpu().detach().numpy()
                )

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
            f1 = f1_score(
                targets_for_head, predictions_for_head, average="macro", zero_division=0
            )
            if self.run:
                log_path = "/".join(["metrics/epoch/", head_name])
                self.run[log_path].log(head_report)
                self.run[f"/metrics/epoch/f1/{head_name}"].log(f1)

        if self.trainer.global_step > 0:
            print("Saving model...")
            torch.save(self.state_dict(), self.model_save_dir)

            self.scheduler.step()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.lr,
            max_lr=0.001,
            step_size_up=50,
            mode="exp_range",
            gamma=0.85,
            cycle_momentum=False,
        )
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
            self.run["params/lr"].log(optimizer.param_groups[0]["lr"])
