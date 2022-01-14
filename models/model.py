import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import MultiplicativeLR
from models.model_loader import ModelLoader
from models.feature_extractors.multi_frame_feature_extractor import (
    MultiFrameFeatureExtractor,
)


class GlossTranslationModel(pl.LightningModule):
    """Awesome model for Gloss Translation"""

    def __init__(
        self,
        lr=1e-5,
        multiply_lr_step=0.7,
        warmup_steps=100.0,
        transformer_output_size=1024,
        representation_size=2048,
        num_classes=1000,
        feature_extractor_name="cnn_extractor",
        transformer_name="vanilla_transformer",
        model_save_dir="",
    ):
        super().__init__()

        # parameters
        self.lr = lr
        self.model_save_dir = model_save_dir
        self.warmup_steps = warmup_steps
        self.multiply_lr_step = multiply_lr_step

        # losses
        self.ce_loss = nn.CrossEntropyLoss()

        # models-parts
        self.model_loader = ModelLoader()
        self.feature_extractor = self.model_loader.load_feature_extractor(
            feature_extractor_name, representation_size
        )
        self.multi_frame_feature_extractor = MultiFrameFeatureExtractor(
            self.feature_extractor
        )
        self.transformer = self.model_loader.load_transformer(
            transformer_name, representation_size, transformer_output_size
        )
        self.cls_head = nn.Linear(transformer_output_size, num_classes)

    def forward(self, input, **kwargs):
        x = self.multi_frame_feature_extractor(input)
        x = self.transformer(x)
        for i in range(0, len(self.cls_head)):
            predictions.append(self.cls_head[i](x.to("cpu")))   
        return predictions

    def training_step(self, batch, batch_idx):
        losses=[]
        input, target = batch
        predictions = self(input)
        for i in range(0, len(predictions)):
            losses.append(self.ce_loss(predictions[i].to("cpu"), target[i].to("cpu")))
        loss = np.sum(losses)
        self.log("metrics/batch/training_loss", loss, prog_bar=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        losses=[]
        input, target = batch
        predictions = self(input)
        for i in range(0, len(predictions)):
            losses.append(self.ce_loss(predictions[i].to("cpu"), target[i].to("cpu")))
        loss = np.sum(losses)
        self.log("metrics/batch/validation_loss", loss)

    def validation_epoch_end(self, out):
        # TO-DO validation metrics at the epoch end
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
        self.log("params/lr", self.lr, prog_bar=False)
