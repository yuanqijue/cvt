import torch
import torch.nn as nn
from timm.models.layers.helpers import to_2tuple
from functools import partial, reduce


class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0, K=4096, pretrained=True):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        K: queue size of negative samples(default: 4096)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.T = T

        # build encoders
        self.base_encoder = base_encoder(num_classes=dim, pretrained=pretrained)
        self.momentum_encoder = base_encoder(num_classes=dim)

        # projectors
        self.projector = MLP(3, dim, mlp_dim, dim)
        self.momentum_projector = MLP(3, dim, mlp_dim, dim)

        # predictor
        self.predictor = MLP(2, dim, mlp_dim, dim)

        if hasattr(base_encoder, 'patch_embed'):
            self.base_encoder.patch_embed = ConvStem(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
            self.momentum_encoder.patch_embed = ConvStem(img_size=224, patch_size=16, in_chans=3, embed_dim=768)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.projector.parameters(), self.momentum_projector.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        if torch.cuda.is_available():
            nn.SyncBatchNorm.convert_sync_batchnorm(self.base_encoder)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.momentum_projector)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.projector)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.momentum_projector)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)

        # create the queue
        self.register_buffer("queue1", torch.randn(dim, K))
        self.register_buffer("queue2", torch.randn(dim, K))
        self.queue1 = nn.functional.normalize(self.queue1, dim=0)
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys1, keys2):
        # gather keys before updating queue
        keys1 = concat_all_gather(keys1)
        keys2 = concat_all_gather(keys2)

        batch_size = keys1.shape[0]

        ptr = int(self.queue_ptr)

        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue1[:, ptr:ptr + batch_size] = keys1.T
        self.queue2[:, ptr:ptr + batch_size] = keys2.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

        for param_b, param_m in zip(self.projector.parameters(), self.momentum_projector.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k, queue):

        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        if torch.cuda.is_available():
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        else:
            labels = torch.zeros(logits.shape[0], dtype=torch.long)

        return nn.functional.cross_entropy(logits, labels)

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        q1 = self.predictor(self.projector(self.base_encoder(x1)))
        q2 = self.predictor(self.projector(self.base_encoder(x2)))
        q1 = nn.functional.normalize(q1, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_projector(self.momentum_encoder(x1))
            k2 = self.momentum_projector(self.momentum_encoder(x2))
            k1 = nn.functional.normalize(k1, dim=1)
            k2 = nn.functional.normalize(k2, dim=1)

        # compute loss
        loss = self.contrastive_loss(q1, k2, self.queue2) + self.contrastive_loss(q2, k1, self.queue1)
        self._dequeue_and_enqueue(k1, k2)
        return loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if torch.cuda.is_available():
        tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
    else:
        output = tensor
    return output


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        super(MLP, self).__init__()
        mlp = []
        for layer in range(num_layers):
            dim1 = input_dim if layer == 0 else mlp_dim
            dim2 = output_dim if layer == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if layer < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        self.mlp_linear = nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp_linear(x)


class ConvStem(nn.Module):
    """
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 16, 'ConvStem only supports patch size of 16'
        assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for ConvStem'

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(4):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2

        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[
            1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
