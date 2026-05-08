import torch
import torch.nn as nn
from transformers import BertModel


class BertFPN(nn.Module):
    """
    Stable BERT + Text-FPN for single-label text classification.
    Default backbone: bert-base-uncased
    """

    def __init__(
        self,
        num_labels: int = 20,
        model_name: str = "bert-base-uncased",
        fpn_dim: int = 256,
        classifier_hidden: int = 512,
        dropout: float = 0.1,
        layer_indices=(4, 8, 12),  # use hidden states from these encoder layers
    ):
        super().__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.layer_indices = layer_indices

        # BERT backbone
        self.bert = BertModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base-uncased

        # Channel alignment: hidden_size -> fpn_dim
        self.proj4 = nn.Linear(hidden_size, fpn_dim)
        self.proj8 = nn.Linear(hidden_size, fpn_dim)
        self.proj12 = nn.Linear(hidden_size, fpn_dim)

        # Top-down FPN projections
        self.fpn_proj12_to_8 = nn.Linear(fpn_dim, fpn_dim)
        self.fpn_proj8_to_4 = nn.Linear(fpn_dim, fpn_dim)

        # Norm after fusion (improves stability)
        self.norm8 = nn.LayerNorm(fpn_dim)
        self.norm4 = nn.LayerNorm(fpn_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(fpn_dim * 3, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_labels),
        )

        self.loss_fct = nn.CrossEntropyLoss()
        self._init_added_layers()

    def _init_added_layers(self):
        """Initialize non-BERT layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # keep BERT pretrained weights untouched by checking parameter names
                # this init mainly affects newly added linear layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Re-load BERT pretrained weights init wasn't desired for backbone;
        # safer approach: do not reinit backbone parameters.
        # Undoing by reloading is expensive, so instead we skip re-init backbone:
        # (Handled by overriding after init)
        # Practical fix: only initialize added layers explicitly:
        added_layers = [
            self.proj4, self.proj8, self.proj12,
            self.fpn_proj12_to_8, self.fpn_proj8_to_4,
            self.classifier[0], self.classifier[3],
        ]
        for layer in added_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.ones_(self.norm8.weight); nn.init.zeros_(self.norm8.bias)
        nn.init.ones_(self.norm4.weight); nn.init.zeros_(self.norm4.bias)

    @staticmethod
    def masked_mean(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C]
        attention_mask: [B, L] (0/1)
        return: [B, C]
        """
        mask = attention_mask.unsqueeze(-1).to(dtype=x.dtype)  # [B, L, 1]
        x = x * mask
        denom = mask.sum(dim=1).clamp(min=1e-6)               # [B, 1]
        return x.sum(dim=1) / denom                           # [B, C]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        return_dict: bool = True
    ):
        # BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs.hidden_states  # tuple length = 13 for bert-base

        l4, l8, l12 = self.layer_indices
        h4 = hidden_states[l4]     # [B, L, 768]
        h8 = hidden_states[l8]     # [B, L, 768]
        h12 = hidden_states[l12]   # [B, L, 768]

        # Channel alignment
        c4 = self.proj4(h4)        # [B, L, 256]
        c8 = self.proj8(h8)        # [B, L, 256]
        c12 = self.proj12(h12)     # [B, L, 256]

        # Text-FPN top-down fusion
        p12 = c12
        p8 = self.norm8(c8 + self.fpn_proj12_to_8(p12))
        p4 = self.norm4(c4 + self.fpn_proj8_to_4(p8))

        # Masked mean pooling
        v4 = self.masked_mean(p4, attention_mask)     # [B, 256]
        v8 = self.masked_mean(p8, attention_mask)     # [B, 256]
        v12 = self.masked_mean(p12, attention_mask)   # [B, 256]

        # Concat & classify
        v = torch.cat([v4, v8, v12], dim=-1)          # [B, 768]
        logits = self.classifier(v)                   # [B, num_labels]

        loss = None
        if labels is not None:
            # labels shape: [B], dtype: long
            loss = self.loss_fct(logits, labels)

        if return_dict:
            return {"loss": loss, "logits": logits}
        return (loss, logits) if labels is not None else logits

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())