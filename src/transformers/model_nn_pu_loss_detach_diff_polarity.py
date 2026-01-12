import torch
import torch.nn as nn
import lightning.pytorch as pl
from depthcharge.transformers import SpectrumTransformerEncoder
from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision
from torchmetrics.classification import BinaryF1Score
from depthcharge.encoders import FloatEncoder


# --- 1. The Custom nnPU Loss Class ---


class NNPULoss(nn.Module):
    def __init__(self, beta=0.0, gamma=1.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets, priors):
        # targets: 1 = Positive, 0 = Unlabeled
        # priors: shape (batch_size, 1)

        positive_indices = (targets == 1).view(-1)
        unlabeled_indices = (targets == 0).view(-1)

        if positive_indices.sum() == 0 or unlabeled_indices.sum() == 0:
            return self.loss_func(logits, targets).mean()

        # Calculate raw losses for all samples
        l_p_plus = self.loss_func(logits, torch.ones_like(logits))
        l_p_minus = self.loss_func(logits, torch.zeros_like(logits))
        l_u_minus = self.loss_func(logits, torch.zeros_like(logits))

        # Risk for Positive samples: pi * L(f(x), 1)
        # We take the mean over the positive samples in the batch
        r_p_plus = (priors[positive_indices] * l_p_plus[positive_indices]).mean()

        # Risk for Unlabeled samples: L(f(x), 0) - pi * L(f(x), 0)
        # r_u_minus is the average loss on the unlabeled set
        r_u_minus = l_u_minus[unlabeled_indices].mean()

        # r_p_minus is the expected loss of positives treated as negatives
        r_p_minus = (priors[positive_indices] * l_p_minus[positive_indices]).mean()

        negative_risk = r_u_minus - r_p_minus

        # Non-negative constraint
        if negative_risk < -self.beta:
            # The "Chainer Trick" logic to keep gradients flowing while clipping value
            target_val = r_p_plus - self.beta
            gradient_signal = -self.gamma * negative_risk
            return gradient_signal + (target_val - gradient_signal).detach()
        else:
            return r_p_plus + negative_risk


# --- 2. Your Model (Unchanged) ---
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


# --- 3. The Lightning Module (Updated) ---
class SimpleSpectraTransformer(pl.LightningModule):
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
        prior_pos: float = 0.50,
        prior_neg: float = 0.35,
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
        self.prior_pos = prior_pos
        self.prior_neg = prior_neg

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
        # Note: Sigmoid is handled implicitly by loss, but used for metrics

        # --- REPLACED: Use Custom PU Loss ---
        self.loss_fn = NNPULoss()

        # --- METRICS ---
        # We only keep Recall (Sensitivity) because we know P labels are real.
        # Accuracy/Precision are misleading on PU validation sets.
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

        # Extract polarity from instrument_settings
        inst_settings = batch["instrument_settings"]
        is_positive_polarity = (inst_settings[:, 15] > 0.5).float().view(-1, 1)

        # Create a tensor of priors for this specific batch
        # prior = is_pos * prior_pos + (1 - is_pos) * prior_neg
        batch_priors = (
            is_positive_polarity * self.prior_pos + (1 - is_positive_polarity) * self.prior_neg
        )

        loss = self.loss_fn(logits.view(-1, 1), targets, batch_priors)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).int()

        return loss, preds, targets, logits

    def training_step(self, batch, batch_idx):
        loss, preds, targets, logits = self.step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # Log recall (Are we finding the known positives?)
        self.train_accuracy.update(preds, targets)
        self.train_precision.update(preds, targets)
        self.train_f1.update(preds, targets)
        self.train_recall.update(preds, targets)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, preds, targets, logits = self.step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # Log recall (Are we finding the known positives?)
        self.val_accuracy.update(preds, targets)
        self.val_precision.update(preds, targets)
        self.val_f1.update(preds, targets)
        self.val_recall.update(preds, targets)

        return {"loss": loss}

    def on_train_epoch_end(self):
        """Reset training metrics at the end of each epoch."""
        self.log("train_accuracy", self.train_accuracy.compute(), prog_bar=True)
        self.log("train_precision", self.train_precision.compute())
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.log("train_recall", self.train_recall.compute(), prog_bar=True)
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_f1.reset()
        self.train_recall.reset()

    def on_validation_epoch_end(self):
        """Reset validation metrics at the end of each epoch."""
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
