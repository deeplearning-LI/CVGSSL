import torch
import torch.nn as nn
import torch.nn.functional as F

activation = {}

def get_activation(name):
    """Register hook to extract intermediate activations."""
    def hook(model, input, output):
        activation[name] = output
    return hook


class LabelSmoothingCELoss(nn.Module):
    """CrossEntropy loss with label smoothing."""
    def __init__(self, label_smooth: float = None, class_num: int = 6):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.label_smooth is not None:
            logits = F.log_softmax(pred, dim=1)
            target = F.one_hot(target, self.class_num).float()
            target = (1.0 - self.label_smooth) * target + self.label_smooth / self.class_num
            loss = -torch.sum(target * logits, dim=1)
            return loss.mean()
        else:
            return F.cross_entropy(pred, target)


class CVGSSL(nn.Module):
    """CLIP-V Guide Self-supervised Learning."""
    def __init__(self, model: nn.Module, out_dim: int = 128, mlp_dim: int = 4096, T1: float = 1.0):
        super().__init__()
        self.T1 = T1
        self.T2 = 0.2
        self.margin = 0.2

        self.base_encoder = model
        # ‼️Please make adjustments according to the dimension of the last layer of the base model.
        self.clip_projector = self._build_mlp(3, 512, 1024, out_dim)
        self._build_projector_and_predictor_mlps(out_dim, mlp_dim)

    def _build_mlp(self, num_layers: int, input_dim: int, mlp_dim: int, output_dim: int, last_bn: bool = True) -> nn.Sequential:
        """Build MLP with optional BatchNorm and ReLU."""
        layers = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim
            layers.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                layers.append(nn.BatchNorm1d(dim2))
                layers.append(nn.ReLU(inplace=True))
            elif last_bn:
                layers.append(nn.BatchNorm1d(dim2, affine=True))
        return nn.Sequential(*layers)

    def _build_projector_and_predictor_mlps(self, dim: int, mlp_dim: int):
        """To be overridden by derived classes."""
        pass

    def contrastive_loss(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Standard contrastive loss."""
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        logits = torch.einsum('nc,mc->nm', q, k) / self.T1
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels) * (2 * self.T1)

    def encourage_same(self, q: list) -> torch.Tensor:
        """Encourage consistency among sequential projections."""
        loss = 0
        for i in range(len(q)):
            q1 = F.normalize(q[i], dim=1)
            q2 = F.normalize(q[(i + 1) % len(q)], dim=1)
            logits = torch.einsum('nc,nc->n', q1, q2).unsqueeze(0)
            label = torch.argmax(logits, dim=1)
            loss += F.cross_entropy(logits, label)
        return loss / len(q)

    def KL_loss(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """KL divergence between similarity distributions (not used in forward)."""
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        v = F.normalize(v, dim=1)

        logits1 = torch.einsum('nc,mc->nc', q, k) / self.T1
        logits2 = torch.einsum('nc,mc->nc', k, v) / self.T1

        return F.kl_div(F.log_softmax(logits1, dim=1), F.softmax(logits2, dim=1), reduction='batchmean')

    def variance_euc_loss(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Euc distance for variance-based supervision (optional)."""
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        return torch.einsum('nc,nc->n', q, k)

    def forward(self, x1: torch.Tensor, x2: list, clip_image_feature: list, epoch: int) -> torch.Tensor:
        """Main forward logic combining contrastive and distillation losses."""
        base_features = self.base_encoder(x1)
        base_predictor = self.predictor(base_features)

        clip_projections = []
        loss_distill = 0
        constra_loss = 0

        for i in range(len(clip_image_feature)):
            if i < len(clip_image_feature) - 1:
                other_features = self.base_encoder(x2[i])
                other_predictor = self.predictor(other_features)
                constra_loss += self.contrastive_loss(other_predictor, base_features.detach())
                constra_loss += self.contrastive_loss(base_features.detach(), base_predictor)

            clip_proj = self.clip_projector(clip_image_feature[i])
            clip_projections.append(clip_proj)
            loss_distill += self.contrastive_loss(base_predictor, clip_proj.detach())
            loss_distill += self.contrastive_loss(base_predictor.detach(), clip_proj)

        encourage_loss = self.encourage_same(clip_projections)

        return loss_distill / len(clip_image_feature) + constra_loss / len(x2) + 0.01 * encourage_loss


class CVGSSL_ViT(CVGSSL):
    """ViT backbone adaptation for CVGSSL."""
    def _build_projector_and_predictor_mlps(self, dim: int, mlp_dim: int):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


class CVGSSL_ResNet(CVGSSL):
    """ResNet backbone adaptation for CVGSSL."""
    def _build_projector_and_predictor_mlps(self, dim: int, mlp_dim: int):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, last_bn=False)
