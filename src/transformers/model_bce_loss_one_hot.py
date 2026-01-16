import torch
import torch.nn as nn
import lightning.pytorch as pl
from depthcharge.transformers import SpectrumTransformerEncoder
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision
from torchmetrics.classification import BinaryF1Score, BinaryRecall
from depthcharge.encoders import FloatEncoder


class MyModel(SpectrumTransformerEncoder):
    """Our custom model class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.precursor_mz_encoder = FloatEncoder(self.d_model)
        self.apply(self.init_weights)

    def global_token_hook(self, mz_array, intensity_array, precursor_mz=None, *args, **kwargs):
        if precursor_mz is None:
            raise ValueError("precursor_mz must be provided in the batch.")
        precursor_mz = precursor_mz.type_as(mz_array).view(-1, 1)
        mz_rep = self.precursor_mz_encoder(precursor_mz)
        if mz_rep.dim() == 3 and mz_rep.shape[1] == 1:
            mz_rep = mz_rep.squeeze(1)
        return mz_rep

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=1)


class SimpleSpectraTransformer(pl.LightningModule):
    """Model using SpectrumTransformerEncoder for MS1 spectra and linear layers for acquisition parameters."""

    def __init__(
        self,
        d_model,
        n_layers,
        dropout,
        lr=0.001,
        hidden_fc1: int = 64,
        instrument_embedding_dim: int = 16,
        encoder_lr: float = 1e-4,
        linear_lr: float = 1e-3,
        weight_decay: float = 1e-3,
        optimizer_name: str = "AdamW",
        validation_threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.d_model = d_model
        self.n_layers = n_layers
        self.lr = float(lr)
        self.encoder_lr = float(encoder_lr)
        self.linear_lr = float(linear_lr)
        self.weight_decay = float(weight_decay)
        self.optimizer_name = optimizer_name
        self.validation_step_outputs = []

        # Transformer encoder
        self.spectrum_encoder = MyModel(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Instrument embeddings
        self.fc_instrument_1 = nn.Linear(22, hidden_fc1)
        self.dropout_instrument = nn.Dropout(dropout)
        self.fc_instrument_2 = nn.Linear(hidden_fc1, instrument_embedding_dim)

        # Combined feature projection
        self.fc_combined = nn.Linear(d_model + instrument_embedding_dim, d_model)
        self.dropout_combined = nn.Dropout(dropout)

        # Output
        self.fc_output = nn.Linear(d_model, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        #  BCEWithLogitsLoss
        self.bce_loss = nn.BCEWithLogitsLoss()
        # Metrics
        self.train_accuracy = BinaryAccuracy()
        self.train_precision = BinaryPrecision()
        self.val_accuracy = BinaryAccuracy()
        self.val_precision = BinaryPrecision()
        self.train_f1 = BinaryF1Score()
        self.train_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()
        self.val_recall = BinaryRecall()
        self.validation_threshold = validation_threshold

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, batch):
        mz_array = batch["mz"].float()
        intensity_array = batch["intensity"].float()
        precursor_mz = batch["precursor_mz"].float()
        spectra_emb, _ = self.spectrum_encoder(mz_array, intensity_array, precursor_mz=precursor_mz)
        spectra_emb = spectra_emb[:, 0, :]

        instrument_settings = batch["instrument_settings"].float()
        instrument_emb = self.relu(self.fc_instrument_1(instrument_settings))
        instrument_emb = self.dropout_instrument(instrument_emb)
        instrument_emb = self.relu(self.fc_instrument_2(instrument_emb))

        combined_emb = torch.cat((spectra_emb, instrument_emb), dim=-1)
        combined_emb = self.relu(self.fc_combined(combined_emb))
        combined_emb = self.dropout_combined(combined_emb)

        output = self.fc_output(combined_emb)
        return output

    def step(self, batch):
        logits = self(batch)
        targets = batch["labels"].float().view(-1, 1)
        loss = self.bce_loss(logits.view(-1, 1), targets)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).int()
        return loss, preds, targets, logits

    def training_step(self, batch, batch_idx):
        loss, preds, targets, logits = self.step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_accuracy.update(preds, targets)
        self.train_precision.update(preds, targets)
        self.train_f1.update(preds, targets)
        self.train_recall.update(preds, targets)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, preds, targets, logits = self.step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_accuracy.update(preds, targets)
        self.val_precision.update(preds, targets)
        self.val_f1.update(preds, targets)
        self.val_recall.update(preds, targets)
        return {"loss": loss, "logits": logits, "labels": targets.view(-1, 1)}

        # Add this line:

    def on_train_epoch_end(self):
        self.log("train_accuracy", self.train_accuracy.compute(), prog_bar=True)
        self.log("train_precision", self.train_precision.compute())
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.log("train_recall", self.train_recall.compute(), prog_bar=True)
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_f1.reset()
        self.train_recall.reset()

    def on_validation_epoch_start(self):
        # Clear the list at the start of every validation epoch
        self.validation_step_outputs.clear()

    def on_validation_epoch_end(self):
        self.log("val_accuracy", self.val_accuracy.compute(), prog_bar=True)
        self.log("val_precision", self.val_precision.compute())
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.log("val_recall", self.val_recall.compute(), prog_bar=True)
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_f1.reset()
        self.val_recall.reset()

    def configure_optimizers(self):
        wd = self.weight_decay
        param_groups = [
            {
                "params": self.spectrum_encoder.parameters(),
                "lr": self.encoder_lr,
                "weight_decay": wd,
            },
            {"params": self.fc_output.parameters(), "lr": self.linear_lr, "weight_decay": wd},
            {"params": self.fc_combined.parameters(), "lr": self.linear_lr, "weight_decay": wd},
            {"params": self.fc_instrument_1.parameters(), "lr": self.linear_lr, "weight_decay": wd},
            {"params": self.fc_instrument_2.parameters(), "lr": self.linear_lr, "weight_decay": wd},
        ]
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(param_groups, eps=1e-4)
        else:
            optimizer = torch.optim.AdamW(param_groups, eps=1e-4)
        return optimizer
