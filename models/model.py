import neptune.new as neptune
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from config import NEPTUNE_API_TOKEN, NEPTUNE_PROJECT_NAME
from sklearn.metrics import classification_report, f1_score
from torch.optim.lr_scheduler import MultiplicativeLR
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
        lr=1e-5,
        multiply_lr_step=0.7,
        warmup_steps=100.0,
        warmup_start_lr=1e-6,
        transformer_output_size=1024,
        representation_size=2048,
        feedforward_size=4096,
        num_encoder_layers=1,
        num_segments=8,
        num_attention_heads=16,
        transformer_dropout_rate=0.1,
        classification_mode="gloss",
        feature_extractor_name="cnn_extractor",
        feature_extractor_model_path="efficientnet_b1",
        transformer_name="fake_transformer",
        model_save_dir="",
        neptune=False,
        classification_heads={"gloss": {
                                "num_class": 2400, 
                                "loss_weight": 1}
                            },
        freeze_scheduler=None,
    ):
        super().__init__()

        if neptune:
            tags = [classification_mode, feature_extractor_name, transformer_name]
            self.run = initialize_neptun(tags)
            self.run["parameters"] = {
                "lr": lr,
                "multiply_lr_step": multiply_lr_step,
                "warmup_steps": warmup_steps,
                "warmup_start_lr": warmup_start_lr,
                "transformer_output_size": transformer_output_size,
                "representation_size": representation_size,
                "feedforward_size": feedforward_size,
                "num_encoder_layers": num_encoder_layers,
                "num_segments": num_segments,
                "num_attention_heads": num_attention_heads,
                "transformer_dropout_rate": transformer_dropout_rate,
                "classification_mode": classification_mode,
                "feature_extractor_name": feature_extractor_name,
                "feature_extractor_model_path": feature_extractor_model_path,
                "transformer_name": transformer_name,
                "classification_heads": classification_heads,
            }
        else:
            self.run = None

        # parameters
        self.lr = lr
        self.model_save_dir = model_save_dir
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.multiply_lr_step = multiply_lr_step
        self.classification_heads = classification_heads
        self.cls_head_internal_layers = []
        self.cls_head_final_layer = []
        self.loss_weights = []
        for value in self.classification_heads.values():
            self.cls_head_internal_layers.append(nn.Sequential(
                nn.Linear(transformer_output_size, transformer_output_size),
                nn.ELU(0.1),
                nn.Dropout(transformer_dropout_rate),
                nn.Linear(transformer_output_size, transformer_output_size),
                nn.ELU(0.1),
                nn.Dropout(transformer_dropout_rate),
            ))
            self.cls_head_final_layer.append(nn.Linear(transformer_output_size, value["num_class"]))
            self.loss_weights.append(value["loss_weight"])

        # losses
        def custom_loss(*args, **kwargs):
            return nn.CrossEntropyLoss(reduction="sum", *args, **kwargs)
        self.summary_loss = SummaryLoss(custom_loss, self.loss_weights)

        # models-parts
        self.model_loader = ModelLoader()
        self.feature_extractor = self.model_loader.load_feature_extractor(
            feature_extractor_name=feature_extractor_name,
            representation_size = representation_size,
            model_path=feature_extractor_model_path,
        )
        self.multi_frame_feature_extractor = MultiFrameFeatureExtractor(
            self.feature_extractor
        )
        if transformer_name == "sign_language_transformer":
            self.transformer = self.model_loader.load_transformer(
                transformer_name,
                input_size=representation_size,
                output_size=transformer_output_size,
                feedforward_size=feedforward_size,
                num_encoder_layers=num_encoder_layers,
                num_frames=num_segments,
                num_attention_heads=num_attention_heads,
                dropout_rate=transformer_dropout_rate,
            )
        else:
            self.transformer = self.model_loader.load_transformer(
                transformer_name, representation_size, transformer_output_size
            )

        self.freeze_scheduler = freeze_scheduler
        if self.freeze_scheduler is not None:
            self.configure_freeze_scheduler()

    def forward(self, input, **kwargs):
        predictions = []
        x = self.multi_frame_feature_extractor(input.to(self.device))
        x = self.transformer(x)
        for i in range(len(self.cls_head_final_layer)):
            internal_layers = self.cls_head_internal_layers[i]
            final_layer = self.cls_head_final_layer[i]
            y = internal_layers(x.cpu())
            predictions.append(final_layer(y))
        return predictions

    def training_step(self, batch, batch_idx):
        input, target = batch
        targets = target["target"]
        predictions = self(input)
        loss = self.summary_loss(predictions, targets)
        if (self.freeze_scheduler is not None) and self.freeze_scheduler["freeze_mode"] == "step":
            self.freeze_step()
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

        for nr_head, head_targets in enumerate(all_targets):
            head_name = head_names[nr_head]
            report = classification_report(all_targets[nr_head],
                                           all_predictions[nr_head],
                                           zero_division=0)
            head_report = "\n".join(
                [
                    head_name,
                    report,
                ]
            )
            f1 = f1_score(all_targets[nr_head], all_predictions[nr_head],
                          average='macro', zero_division=0)
            print(head_report)
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
            new_lr = min(self.lr,
                         self.warmup_start_lr +
                         (self.lr - self.warmup_start_lr) * float(self.trainer.global_step) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr

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
