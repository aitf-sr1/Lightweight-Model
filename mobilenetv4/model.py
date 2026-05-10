import torch
import torch.nn as nn
import timm

from config import IMG_SIZE, N_LABELS, TIMM_MODELS, DROPOUT, MODEL_SIZE

class ImageModel(nn.Module):

    def __init__(self, n_labels: int = N_LABELS, dropout: float = DROPOUT,
                 model_size: str = MODEL_SIZE):
        super().__init__()
        timm_name = TIMM_MODELS[model_size]
        self.backbone = timm.create_model(timm_name, pretrained=True,
                                          num_classes=0, global_pool='')

        self.backbone.eval()
        with torch.no_grad():
            dummy    = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            feat_map = self.backbone(dummy)
            feat_dim = feat_map.shape[1]
        print(f"[Model] {timm_name}  feat_dim={feat_dim}  "
              f"spatial={feat_map.shape[2]}×{feat_map.shape[3]}")

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_labels),


        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(img)
        pool = self.pool(feat)
        return self.head(pool)

def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
