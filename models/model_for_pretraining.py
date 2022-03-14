import neptune.new as neptune
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
            lr=1e-5,
            multiply_lr_step=0.7,
            warmup_steps=100.0,
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
            steps_per_epoch=1000
    ):
        super().__init__()

        if neptune:
            tags = [classification_mode,
                    feature_extractor_name,
                    transformer_name,
                    "pre-training"]
            self.run = initialize_neptun(tags)
            self.run["parameters"] = {
                "lr": lr,
                "multiply_lr_step": multiply_lr_step,
                "warmup_steps": warmup_steps,
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
        self.multiply_lr_step = multiply_lr_step
        self.classification_heads = classification_heads
        self.cls_head = []
        self.loss_weights = []
        for value in self.classification_heads.values():
            self.cls_head.append(nn.Linear(representation_size, value["num_class"]))
            self.loss_weights.append(value["loss_weight"])

        # losses
        self.summary_loss = SummaryLoss(nn.CrossEntropyLoss, self.loss_weights)

        # models-parts
        self.model_loader = ModelLoader()
        self.feature_extractor = self.model_loader.load_feature_extractor(
            feature_extractor_name,
            representation_size,
            model_path=feature_extractor_model_path,
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        steps_per_epoch = self.steps_per_epoch // self.trainer.accumulate_grad_batches
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, epochs=self.trainer.max_epochs,
        #                                                      steps_per_epoch=steps_per_epoch)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_per_epoch // 10,
                                                                    eta_min=self.lr / 100)
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
