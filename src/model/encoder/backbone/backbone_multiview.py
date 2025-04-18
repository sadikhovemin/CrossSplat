import torch
from einops import rearrange
import torch.nn.functional as F

from .unimatch.backbone import CNNEncoder
from .multiview_transformer import MultiViewFeatureTransformer
from .unimatch.utils import split_feature, merge_splits
from .unimatch.position import PositionEmbeddingSine

from ..costvolume.conversions import depth_to_relative_disparity
from ....geometry.epipolar_lines import get_depth


def feature_add_position_list(features_list, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        features_splits = [
            split_feature(x, num_splits=attn_splits) for x in features_list
        ]

        position = pos_enc(features_splits[0])
        features_splits = [x + position for x in features_splits]

        out_features_list = [
            merge_splits(x, num_splits=attn_splits) for x in features_splits
        ]

    else:
        position = pos_enc(features_list[0])

        out_features_list = [x + position for x in features_list]

    return out_features_list


class DINOFeatureExtractor(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.dino_model = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitl14',
            pretrained=True
        ).to(device).eval()

        for p in self.dino_model.parameters():
            p.requires_grad = False

        self.embed_dim = list(self.dino_model.parameters())[-1].shape[0]

    def forward(self, images):
        # images: (B, V, 3, H, W)
        B, V, C, H, W = images.shape
        x = rearrange(images, 'b v c h w -> (b v) c h w')
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        with torch.no_grad():
            emb = self.dino_model(x)  # (B*V, E)
        return emb.view(B, V, -1)


class BackboneMultiview(torch.nn.Module):
    """docstring for BackboneMultiview."""

    def __init__(
        self,
        feature_channels=128,
        num_transformer_layers=6,
        ffn_dim_expansion=4,
        no_self_attn=False,
        no_cross_attn=False,
        num_head=1,
        no_split_still_shift=False,
        no_ffn=False,
        global_attn_fast=True,
        downscale_factor=8,
        use_epipolar_trans=False,
        use_dino=True
    ):
        super(BackboneMultiview, self).__init__()
        self.feature_channels = feature_channels
        # Table 3: w/o cross-view attention
        self.no_cross_attn = no_cross_attn
        # Table B: w/ Epipolar Transformer
        self.use_epipolar_trans = use_epipolar_trans

        self.use_dino = use_dino

        # NOTE: '0' here hack to get 1/4 features
        self.backbone = CNNEncoder(
            output_dim=feature_channels,
            num_output_scales=1 if downscale_factor == 8 else 0,
        )

        if self.use_dino:
            self.dino_handler = DINOFeatureExtractor(device="cuda")
            dino_dim = self.dino_handler.embed_dim
            self.dino_fuse_proj = torch.nn.Conv2d(
                feature_channels + dino_dim,
                feature_channels,
                kernel_size=1,
                bias=True
            )

        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
            no_cross_attn=no_cross_attn,
        )

    def normalize_images(self, images):
        '''Normalize image to match the pretrained GMFlow backbone.
            images: (B, N_Views, C, H, W)
        '''
        shape = [*[1]*(images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(
            *shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(
            *shape).to(images.device)

        return (images - mean) / std

    def extract_feature(self, images):
        b, v = images.shape[:2]
        concat = rearrange(images, "b v c h w -> (b v) c h w")

        # list of [nB, C, H, W], resolution from high to low
        features = self.backbone(concat)
        if not isinstance(features, list):
            features = [features]
        # reverse: resolution from low to high
        features = features[::-1]

        features_list = [[] for _ in range(v)]
        for feature in features:
            feature = rearrange(feature, "(b v) c h w -> b v c h w", b=b, v=v)
            for idx in range(v):
                features_list[idx].append(feature[:, idx])

        return features_list

    def forward(
        self,
        images,
        attn_splits=2,
        return_cnn_features=False,
        epipolar_kwargs=None,
    ):
        ''' images: (B, N_Views, C, H, W), range [0, 1] '''
        # resolution low to high
        normed = self.normalize_images(images)
        features_list = self.extract_feature(normed)  # list of features

        cur_features_list = [x[0] for x in features_list]

        if self.use_dino:
            dino_emb = self.dino_handler(normed)  # (B, V, E)
            B, V, E = dino_emb.shape
            _, C, Hf, Wf = cur_features_list[0].shape
            dino_feats = dino_emb.view(B, V, E, 1, 1).expand(-1, -1, -1, Hf, Wf)

            fused = []
            for idx, feat in enumerate(cur_features_list):
                df = dino_feats[:, idx]
                cat = torch.cat([feat, df], dim=1)
                fused.append(self.dino_fuse_proj(cat))
            cur_features_list = fused

        if return_cnn_features:
            cnn_features = torch.stack(cur_features_list, dim=1)  # [B, V, C, H, W]

        if self.use_epipolar_trans:
            # NOTE: Epipolar Transformer, only for ablation used
            # we only abalate Epipolar Transformer under 2 views setting
            assert (
                epipolar_kwargs is not None
            ), "must provide camera params to apply epipolar transformer"
            assert len(cur_features_list) == 2, "only use 2 views input for Epipolar Transformer ablation"
            feature0, feature1 = cur_features_list
            epipolar_sampler = epipolar_kwargs["epipolar_sampler"]
            depth_encoding = epipolar_kwargs["depth_encoding"]

            features = torch.stack((feature0, feature1), dim=1)  # [B, V, C, H, W]
            extrinsics = epipolar_kwargs["extrinsics"]
            intrinsics = epipolar_kwargs["intrinsics"]
            near = epipolar_kwargs["near"]
            far = epipolar_kwargs["far"]
            # Get the samples used for epipolar attention.
            sampling = epipolar_sampler.forward(
                features, extrinsics, intrinsics, near, far
            )
            # similar to pixelsplat, use camera distance as position encoding
            # Compute positionally encoded depths for the features.
            collect = epipolar_sampler.collect
            depths = get_depth(
                rearrange(sampling.origins, "b v r xyz -> b v () r () xyz"),
                rearrange(sampling.directions, "b v r xyz -> b v () r () xyz"),
                sampling.xy_sample,
                rearrange(collect(extrinsics), "b v ov i j -> b v ov () () i j"),
                rearrange(collect(intrinsics), "b v ov i j -> b v ov () () i j"),
            )

            # Clip the depths. This is necessary for edge cases where the context views
            # are extremely close together (or possibly oriented the same way).
            depths = depths.maximum(near[..., None, None, None])
            depths = depths.minimum(far[..., None, None, None])
            depths = depth_to_relative_disparity(
                depths,
                rearrange(near, "b v -> b v () () ()"),
                rearrange(far, "b v -> b v () () ()"),
            )
            depths = depth_encoding(depths[..., None])
            target = sampling.features + depths
            source = features

            features = self.transformer((source, target), attn_type="epipolar")
        else:
            # add position to features
            cur_features_list = feature_add_position_list(
                cur_features_list, attn_splits, self.feature_channels)

            # Transformer
            cur_features_list = self.transformer(
                cur_features_list, attn_num_splits=attn_splits)

            features = torch.stack(cur_features_list, dim=1)  # [B, V, C, H, W]

        if return_cnn_features:
            out_lists = [features, cnn_features]
        else:
            out_lists = [features, None]

        return out_lists
