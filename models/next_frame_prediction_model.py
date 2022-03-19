import neptune.new as neptune
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from config import NEPTUNE_API_TOKEN, NEPTUNE_PROJECT_NAME
from torch.optim.lr_scheduler import OneCycleLR

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
        capture_stdout=True,
        capture_stderr=True,
    )


class GlossTranslationModel(pl.LightningModule):
    """Awesome model for Gloss Translation"""

    def __init__(
        self,
        lr=1e-5,
        warmup_steps=100.0,
        transformer_output_size=1024,
        representation_size=2048,
        feedforward_size=4096,
        num_encoder_layers=1,
        num_segments=8,
        num_attention_heads=16,
        transformer_dropout_rate=0.1,
        feature_extractor_name="cnn_extractor",
        feature_extractor_model_path="efficientnet_b3_pruned",
        transformer_name="fake_transformer",
        neptune=False,
        loss_function=nn.TripletMarginLoss,
        *args,
        **kwargs,
    ):
        super().__init__()

        if neptune:
            tags = [feature_extractor_name, feature_extractor_model_path, transformer_name, "pretraining", "key frames"]
            self.run = initialize_neptun(tags)
        else:
            self.run = None

        # parameters
        self.lr = lr
        # losses
        self.loss = loss_function()
        self.num_segments=num_segments
        # models-parts
        self.model_loader = ModelLoader()
        self.multi_frame_feature_extractor = MultiFrameFeatureExtractor(
            self.model_loader.load_feature_extractor(
            feature_extractor_name=feature_extractor_name,
            representation_size = representation_size,
            model_path=feature_extractor_model_path,
        )
        )
        if transformer_name in ["sign_language_transformer", "lstm"]:
            self.transformer = self.model_loader.load_transformer(
                transformer_name,
                input_size=representation_size,
                output_size=transformer_output_size,
                feedforward_size=feedforward_size,
                num_encoder_layers=num_encoder_layers,
                num_frames=num_segments-1,
                dropout_rate=transformer_dropout_rate,
            )
        else:
            self.transformer = self.model_loader.load_transformer(
                transformer_name, representation_size, transformer_output_size
            )
     
    def forward(self, input, **kwargs):
        embeddings = self.multi_frame_feature_extractor(input.to(self.device))
        batch_size = embeddings.shape[0]
        embedding_to_predict = embeddings.reshape(self.num_segments,batch_size,-1)[0].reshape(batch_size,-1)
        embeddings = embeddings.reshape(self.num_segments,batch_size,-1)[:-1].reshape(batch_size,self.num_segments-1,-1)
        predicted_embedding = self.transformer(embeddings).reshape(batch_size,-1)
        return predicted_embedding, embedding_to_predict, embeddings[:,round(self.num_segments/2),:] # could be random

    def training_step(self, batch, batch_idx):
        input, _ = batch
        predicted_embedding, embedding_to_predict, random_embedding = self(input)
        # predict next frame embedding
        # and make each frame embedding less similar than the next one
        loss = self.loss(predicted_embedding, # anchor
                         embedding_to_predict, # positive 
                         random_embedding) # negative
        self.scheduler.step()
        if self.run:
            self.run["metrics/batch/training_loss"].log(loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input, _ = batch
        predicted_embedding, embedding_to_predict, random_embedding = self(input)
        loss = self.loss(predicted_embedding, # anchor
                         embedding_to_predict, # positive 
                         random_embedding) # negative
        if self.run:
            self.run["metrics/batch/validation_loss"].log(loss)
        return {"val_loss": loss}

    def validation_epoch_end(self, out):
        torch.save(self.transformer.state_dict(), "transformer.ckpt")
        torch.save(self.multi_frame_feature_extractor.state_dict(), "multiframe.ckpt")
        

    def configure_optimizers(self):
        # set optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = OneCycleLR(optimizer, max_lr=1e-3,total_steps=80000)
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
