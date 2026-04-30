import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision
from torchmetrics.classification import BinaryF1Score, BinaryRecall
from depthcharge.encoders import FloatEncoder


class NNPULoss(nn.Module):
    def __init__(self, beta=0.0, gamma=1.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.loss_func = nn.BCEWithLogitsLoss(reduction='none')

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


class PrecursorOnlyBaseline(pl.LightningModule):
    def __init__(
            self,
            precursor_embedding_dim: int = 32,
            hidden_dim: int = 64,
            lr: float = 1e-3,
            weight_decay: float = 1e-3,
            optimizer_name: str = "AdamW",
            validation_threshold: float = 0.5,
            prior_pos: float = 0.50,
            prior_neg: float = 0.35,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.optimizer_name = optimizer_name
        self.prior_pos = prior_pos
        self.prior_neg = prior_neg
        self.validation_threshold = validation_threshold


        self.precursor_encoder = FloatEncoder(d_model=precursor_embedding_dim)


        self.mlp = nn.Sequential(
            nn.Linear(precursor_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Output raw logit
        )

        self.loss_fn = NNPULoss()

        self.train_accuracy = BinaryAccuracy()
        self.train_precision = BinaryPrecision()
        self.val_accuracy = BinaryAccuracy()
        self.val_precision = BinaryPrecision()
        self.train_f1 = BinaryF1Score()
        self.train_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()
        self.val_recall = BinaryRecall()

    def forward(self, batch):
        # Extract and format precursor m/z
        precursor_mz = batch["precursor_mz"].float().view(-1, 1)

        # Pass through Depthcharge's FloatEncoder
        prec_emb = self.precursor_encoder(precursor_mz)

        # Flatten sequence dimension if FloatEncoder returns (batch, 1, dim)
        if prec_emb.dim() == 3 and prec_emb.shape[1] == 1:
            prec_emb = prec_emb.squeeze(1)

        # Pass through MLP
        output = self.mlp(prec_emb)
        return output

    def step(self, batch):
        logits = self(batch)
        targets = batch["labels"].float().view(-1, 1)

        # We still extract instrument_settings purely to calculate the batch priors
        # for the NNPULoss, even though the model isn't training on them.
        inst_settings = batch["instrument_settings"]
        is_positive_polarity = (inst_settings[:, 10] > 0.5).float().view(-1, 1)
        batch_priors = is_positive_polarity * self.prior_pos + (1 - is_positive_polarity) * self.prior_neg

        loss = self.loss_fn(logits.view(-1, 1), targets, batch_priors)

        probs = torch.sigmoid(logits)
        preds = (probs >= self.validation_threshold).int()

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

        return {'loss': loss}

    def on_train_epoch_end(self):
        self.log("train_accuracy", self.train_accuracy.compute(), prog_bar=True)
        self.log("train_precision", self.train_precision.compute())
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.log("train_recall", self.train_recall.compute(), prog_bar=True)
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_f1.reset()
        self.train_recall.reset()

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
            {"params": self.precursor_encoder.parameters(), "lr": self.lr, "weight_decay": wd},
            {"params": self.mlp.parameters(), "lr": self.lr, "weight_decay": wd},
        ]
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(param_groups, eps=1e-4)
        else:
            optimizer = torch.optim.AdamW(param_groups, eps=1e-4)
        return optimizer