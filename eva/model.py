import torch
import torch.nn as nn
import timm

from config import IMG_SIZE, N_LABELS, TIMM_NAME, DROPOUT

class ImageModel(nn.Module):
    def __init__(self, n_labels: int = N_LABELS, dropout: float = DROPOUT):
        super().__init__()
        self.backbone = timm.create_model(TIMM_NAME, pretrained=True,
                                          num_classes=0)


        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            self.feat_dim = int(self.backbone(dummy).shape[-1])
        print(f"[Model] backbone feat_dim = {self.feat_dim}")

        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_labels),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(img))

def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
