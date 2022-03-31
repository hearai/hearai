from typing import Dict

import neptune.new as neptune
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from config import NEPTUNE_API_TOKEN, NEPTUNE_PROJECT_NAME
from sklearn.metrics import classification_report, f1_score
from utils.summary_loss import SummaryLoss
from math import ceil

from models.feature_extractors.multi_frame_feature_extractor import (
    MultiFrameFeatureExtractor,
)
from models.model_loader import ModelLoader
from models.common.simple_sequential_model import SimpleSequentialModel
from models.landmarks_models.lanmdarks_sequential_model import LandmarksSequentialModel
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
            loss_function (torch.nn.Module): Loss function.
        """
        super().__init__()

        if general_parameters["neptune"]:
            tags = [train_parameters["classification_mode"], feature_extractor_parameters["name"], transformer_parameters["name"]]
            self.run = initialize_neptun(tags)
            self.run["parameters"] = {
                "general_parameters": general_parameters,
                "train_parameters": train_parameters,
                "feature_extractor_parameters": feature_extractor_parameters,
                "transformer_parameters": transformer_parameters,
                "heads": heads,
                "freeze_scheduler": freeze_scheduler,
                "loss_function": loss_function
            }
        else:
            self.run = None

        # parameters
        self.lr = train_parameters["lr"]
        self.model_save_dir = general_parameters["path_to_save"]
        self.warmup_steps = train_parameters["warmup_steps"]
        self.multiply_lr_step = train_parameters["multiply_lr_step"]
        self.use_frames = train_parameters["use_frames"]
        self.use_landmarks = train_parameters["use_landmarks"]
        self.classification_heads = heads[train_parameters['classification_mode']]
        self.cls_head = nn.ModuleList()
        self.loss_weights = []
        for value in self.classification_heads.values():
            self.cls_head.append(
                HeadClassificationSequentialModel(
                    classes_number=value["num_class"],
                    representation_size=3 * value["num_class"],
                    additional_layers=1,
                    dropout_rate=heads["model"]["dropout_rate"]
                )
            )
            self.loss_weights.append(value["loss_weight"])

        # losses
        self.summary_loss = SummaryLoss(loss_function, self.loss_weights)

        # models-parts
        self.model_loader = ModelLoader()

        representation_size = feature_extractor_parameters["representation_size"]

        self.adjustment_to_representatios_size = nn.LazyLinear(out_features=representation_size)

        if self.use_frames:
            self.multi_frame_feature_extractor = MultiFrameFeatureExtractor(
                self.model_loader.load_feature_extractor(
                    feature_extractor_name=feature_extractor_parameters["name"],
                    representation_size=representation_size,
                    model_path=feature_extractor_parameters["model_path"],
                )
            )
        else:
            self.multi_frame_feature_extractor = None


        self.transformer = self.model_loader.load_transformer(
                transformer_name=transformer_parameters["name"],
                feature_extractor_parameters=feature_extractor_parameters,
                transformer_parameters=transformer_parameters,
                train_parameters=train_parameters
            )

        self.steps_per_epoch = steps_per_epoch
        if freeze_scheduler is not None:
            self.freeze_scheduler = freeze_scheduler
            self.configure_freeze_scheduler()

    def forward(self, input, **kwargs):
        predictions = []
        frames, landmarks = input

        if self.use_frames:
            x = self.multi_frame_feature_extractor(frames.to(self.device))

        if self.use_landmarks:
            x_landmarks = self._prepare_landmarks_tensor(landmarks)
            if self.use_frames:
                x = torch.concat([x, x_landmarks], dim=-1)
            else:
                x = x_landmarks

        x = self.adjustment_to_representatios_size(x)
        x = self.transformer(x)

        for head in self.cls_head:
            predictions.append(head(x))
        return predictions

    def _prepare_landmarks_tensor(self, landmarks):
        concatenated_landmarks = np.concatenate(
            [landmarks[landmarks_name] for landmarks_name in landmarks.keys()],
            axis=-1
        )
        return torch.as_tensor(concatenated_landmarks, dtype=torch.float32, device=self.device)

    def training_step(self, batch, batch_idx):
        targets, predictions, losses = self._process_batch(batch)
        if self.global_step < 2:
            for name, child in self.named_children():
                for param in child.parameters():
                    param.requires_grad = True
        if self.freeze_scheduler["freeze_mode"] == "step":
            self.freeze_step()
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

            self.scheduler.step()
            if (self.freeze_scheduler is not None) and self.freeze_scheduler["freeze_mode"] == "epoch":
                self.freeze_step()

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                             max_lr=self.lr,
                                                             div_factor=100,
                                                             final_div_factor=10,
                                                             pct_start=0.2,
                                                             total_steps=self.trainer.max_epochs * self.steps_per_epoch + 2)
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

    def configure_freeze_scheduler(self):
        ### TO-DO check if all params are correctly set
        # e.g. check if all lists are the same length
        # check if values are bools
        self.freeze_scheduler["current_pattern"] = 0
        self.freeze_scheduler["current_counter"] = 0
        self.freeze_step()

    def freeze_step(self):
        ### TO- DO
        #  If the `freeze_pattern_repeats` is set as an integer isntead of a list, 
        # e.g. `freeze_pattern_repeats = 3`, it is equal to a pattern 
        # `feature_extractor = [True, False] * freeze_pattern_repeats`, 
        # hence it is exactly the same as:
        #  ```
        #  "model_params": {
        #         "feature_extractor":  [True, False, True, False, True, False],
        #         "transformer": [False, True,False, True, False, True],
        #     }
        # ```
        if self.freeze_scheduler is not None:
            self.freeze_update()
            for params_to_freeze in list(self.freeze_scheduler["model_params"].keys()):
                if self.freeze_scheduler["current_pattern"] >= len(
                        self.freeze_scheduler["model_params"][params_to_freeze]
                ):
                    current_pattern = True
                else:
                    current_pattern = self.freeze_scheduler["model_params"][
                        params_to_freeze
                    ][self.freeze_scheduler["current_pattern"]]

                for name, child in self.named_children():
                    if params_to_freeze in name:
                        for param in child.parameters():
                            param.requires_grad = not current_pattern
                if self.freeze_scheduler["verbose"]:
                    print(
                        "Freeze status:",
                        params_to_freeze,
                        "set to",
                        str(current_pattern),
                    )

    def freeze_update(self):
        if self.freeze_scheduler["current_pattern"] >= len(
                self.freeze_scheduler["model_params"][
                    list(self.freeze_scheduler["model_params"].keys())[0]
                ]
        ):
            return
        if (
                self.freeze_scheduler["current_counter"]
                >= self.freeze_scheduler["freeze_pattern_repeats"][
            self.freeze_scheduler["current_pattern"]
        ]
        ):
            self.freeze_scheduler["current_pattern"] += 1
            self.freeze_scheduler["current_counter"] = 0
        self.freeze_scheduler["current_counter"] += 1
