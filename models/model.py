import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiplicativeLR

# model imports...
# loss imports...
from models.model_loader import ModelLoader


class GlossTranslationModel(pl.LightningModule):
    """Awesome model for Gloss Translation"""

    def __init__(self, lr=1e-5,
                 multiply_lr_step=0.7,
                 warmup_steps=100.0,
                 transformer_name="vanilla_trasnformer",
                 feature_extractor_name="cnn_extractor",
                 model_save_dir="path/to/model/save/dir"):
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
        self.feature_extractor = self.model_loader.load_feature_extractor(feature_extractor_name,
                                                                          representation_size=12)
        self.transformer = self.model_loader.load_transformer(transformer_name)
        self.cls_head = nn.Linear(transformer_output_size, num_classes)

    def forward(self, input, **kwargs):
        x = self.feature_extractor(input)
        x = self.transformer(input)
        prediction = self.cls_head(x)
        return prediction

    def training_step(self, batch, batch_idx):
        input, target = batch["input"], batch["target"]
        prediction = self(input)
        loss = self.ce_loss(prediction, target)
        self.log("metrics/batch/training_loss", loss, prog_bar=False)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input, target = batch["input"], batch["target"]
        prediction = self(input)
        loss = self.ce_loss(prediction, target)
        self.log("metrics/batch/validation_loss", loss)

    def validation_epoch_end(self, out):
        # TO-DO validation metrics at the epoch end
        if self.trainer.global_step > 0:
            print('Saving model...')
            torch.save(self.model.state_dict(), self.model_save_dir)

    def configure_optimizers(self):
        # set optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # set scheduler: multiply lr every epoch
        def lambd(epoch): return self.multiply_lr_step

        scheduler = MultiplicativeLR(optimizer, lr_lambda=lambd)
        return [optimizer], [scheduler]

    def optimizer_step(self,
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
