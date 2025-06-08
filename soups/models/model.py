import timm
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch import Tensor


class CustomModel(nn.Module):
    def __init__(
        self,
        image_backbone_model: str,
        text_encoder_model: str,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.text_encoder = SentenceTransformer(text_encoder_model)
        self.text_encoder.eval()

        self.image_backbone = timm.create_model(
            image_backbone_model,
            pretrained=True,
            num_classes=0,
        )
        self.fc = nn.Linear(in_features=1536, out_features=num_classes, bias=True)

    def forward(self, images: Tensor, texts: list[str]) -> Tensor:
        text_features = self.text_encoder.encode(texts)  # (B, 768)
        text_features = torch.from_numpy(text_features).to(device=images.device)
        image_features = self.image_backbone(images)  # (B, 768)
        all_features = torch.concat([image_features, text_features], dim=1)  # (B, 1536)
        logits = self.fc(all_features)
        return logits
