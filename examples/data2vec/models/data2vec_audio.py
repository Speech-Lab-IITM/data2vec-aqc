# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import II

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from fairseq.modules import EMAModule, EMAModuleConfig
from fairseq.data.data_utils import compute_mask_indices
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec import (
    ConvFeatureExtractionModel,
    Wav2Vec2Config,
    TransformerEncoder,
)
from fairseq.modules import (
    GradMultiply,
    LayerNorm,
    GumbelVectorQuantizer,
)
from fairseq.utils import buffered_arange, index_put, is_xla_tensor, index_put_scale
from fast_pytorch_kmeans import KMeans


logger = logging.getLogger(__name__)


@dataclass
class Data2VecAudioConfig(Wav2Vec2Config):

    loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )
    average_top_k_layers: int = field(
        default=8, metadata={"help": "how many layers to average"}
    )

    layer_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False
    batch_norm_target_layer: bool = False
    group_norm_target_layer: bool = False

    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    # when to finish annealing ema decay rate
    ema_anneal_end_step: int = II("optimization.max_update")

    ema_transformer_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer"},
    )
    ema_layers_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer layers"},
    )

    max_update: int = II("optimization.max_update")

    min_target_var: float = field(
        default=0.1, metadata={"help": "stop training if target var falls below this"}
    )
    min_pred_var: float = field(
        default=0.01,
        metadata={"help": "stop training if prediction var falls below this"},
    )


def get_annealed_rate(start, end, curr_step, total_steps):
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


@register_model("data2vec_audio", dataclass=Data2VecAudioConfig)
class Data2VecAudioModel(BaseFairseqModel):
    def __init__(self, cfg: Data2VecAudioConfig):
        super().__init__()
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.extractor_embed = feature_enc_layers[-1][0]

        self.ema = None
        self.embed = cfg.encoder_embed_dim

        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        self.post_extract_proj = nn.Linear(self.extractor_embed, cfg.encoder_embed_dim)

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_before = cfg.mask_channel_before
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.n_negatives = cfg.num_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives

        vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else cfg.final_proj_dim
        self.quantizer = GumbelVectorQuantizer(
                dim=self.extractor_embed,
                num_vars=cfg.latent_vars,
                temp=cfg.latent_temp,
                groups=cfg.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
                weight_proj_depth=cfg.quantizer_depth,
                weight_proj_factor=cfg.quantizer_factor,
        )
        self.project_q = nn.Linear(vq_dim, cfg.final_proj_dim)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.extractor_embed)

        self.final_proj = nn.Linear(self.embed, self.embed)
        self.contr_proj = nn.Linear(self.embed, cfg.final_proj_dim)

        self.num_updates = 0

    def make_ema_teacher(self):
        ema_config = EMAModuleConfig(
            ema_decay=self.cfg.ema_decay,
            ema_fp32=True,
        )
        skip_keys = set()
        if self.cfg.ema_layers_only:
            self.cfg.ema_transformer_only = True
            for k, _ in self.encoder.pos_conv.named_parameters():
                skip_keys.add(f"pos_conv.{k}")

        self.ema = EMAModule(
            self.encoder if self.cfg.ema_transformer_only else self,
            ema_config,
            skip_keys=skip_keys,
        )

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.ema is None and self.final_proj is not None:
            logger.info(f"making ema teacher")
            self.make_ema_teacher()
        elif self.training and self.ema is not None:
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.encoder if self.cfg.ema_transformer_only else self)

        self.num_updates = num_updates

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if self.ema is not None:
            k = prefix + "_ema"
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @classmethod
    def build_model(cls, cfg: Data2VecAudioConfig, task=None):
        """Build a new model instance."""

        return cls(cfg)

    def apply_mask(
        self,
        x,
        padding_mask,
        mask_indices=None,
        mask_channel_indices=None,
    ):
        B, T, C = x.shape

        if self.mask_channel_prob > 0 and self.mask_channel_before:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=1,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                    require_same_masks=self.cfg.require_same_masks,
                    mask_dropout=self.cfg.mask_dropout,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb)
        else:
            mask_indices = None

        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            if mask_channel_indices is None:
                mask_channel_indices = compute_mask_indices(
                    (B, C),
                    None,
                    self.mask_channel_prob,
                    self.mask_channel_length,
                    self.mask_channel_selection,
                    self.mask_channel_other,
                    no_overlap=self.no_mask_channel_overlap,
                    min_space=self.mask_channel_min_space,
                )
                mask_channel_indices = (
                    torch.from_numpy(mask_channel_indices)
                    .to(x.device)
                    .unsqueeze(1)
                    .expand(-1, T, -1)
                )
            x = index_put(x, mask_channel_indices, 0)

        return x, mask_indices

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)


    def compute_preds_scale(self, x, y, negatives, filter_negs, sf=0.3):

        logit_temp=0.1
        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1)
        logits = logits / logit_temp
        logits = logits.type_as(x)

        if is_xla_tensor(logits) or neg_is_pos.any():
            if not hasattr(self, "_inftensor"):
                fillval = -float(2**30)
                self._inftensor = (
                    torch.tensor(fillval).to(x.device)
                    if is_xla_tensor(logits)
                    else float("-inf")
                )
            logits[1:] = index_put(logits[1:], neg_is_pos, self._inftensor)
            
            # applying the scaling factor for the logits corresponding to filtered negatives
            logits[1:] = index_put_scale(logits[1:], filter_negs, sf)

        return logits

    def get_logits(self, logits):
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

    def get_targets(self, x):
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    def sample_negatives(self, y, num, padding_count=None):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        # FIXME: what happens if padding_count is specified?
        cross_high = tsz * bsz
        high = tsz - (padding_count or 0)
        with torch.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"

            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            neg_idxs = neg_idxs + (torch.arange(bsz).unsqueeze(1) * high)
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs


    def forward(
        self,
        source,
        padding_mask=None,
        mask=True,
        features_only=False,
        layer=None,
        mask_indices=None,
        mask_channel_indices=None,
        padding_count=None,
    ):

        # extracting both the source and the augmented audios from source
        source_audios = source[0]
        aug_audios = source[1]

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source_audios) #features
            aug_features = self.feature_extractor(aug_audios)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source_audios) #features
                aug_features = self.feature_extractor(aug_audios)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        aug_features = aug_features.transpose(1, 2)

        features = self.layer_norm(features)
        aug_features = self.layer_norm(aug_features)

        unmasked_features = features.clone()
        aug_unmasked_features = aug_features.clone()

        orig_padding_mask = padding_mask

        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
            aug_features = self.post_extract_proj(aug_features)

        pre_encoder_features = None
        if self.cfg.ema_transformer_only:
            pre_encoder_features = features.clone()
            pre_encoder_aug_features = aug_features.clone()

        features = self.dropout_input(features)
        aug_features = self.dropout_input(aug_features)

        unmasked_features = self.dropout_features(unmasked_features)
        aug_unmasked_features = self.dropout_features(aug_unmasked_features)

        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            )
            # applying the same mask (as used for the original sample features) over the augmented sample features
            # this consistency is needed as we are going to compute cross-contrastive losses as well
            x_aug = index_put(aug_features, mask_indices, self.mask_emb)
            if not is_xla_tensor(x) and mask_indices is not None:
                # tpu-comment: reducing the size in a dynamic way causes
                # too many recompilations on xla.
                orig_mregion = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
                orig_mregion_aug = aug_unmasked_features[mask_indices].view(
                    aug_unmasked_features.size(0), -1, aug_unmasked_features.size(-1)
                )
            else:
                orig_mregion = unmasked_features
                orig_mregion_aug = aug_unmasked_features
        else:
            x = features
            x_aug = aug_features
            orig_mregion = unmasked_features
            orig_mregion_aug = aug_unmasked_features
            mask_indices = None

        if features_only == False:
            x_aug, layer_results = self.encoder(
                x_aug,
                padding_mask=padding_mask,
                layer=layer,
            )

        if features_only:
            x, layer_results = self.encoder(
                x,
                padding_mask=padding_mask,
                layer=layer,
            )
            return {
                "x": x,
                "padding_mask": padding_mask,
                "layer_results": layer_results,
            }

        result = {
            "losses": {},
        }


        q = self.quantizer(orig_mregion, produce_targets=False)
        orig_mregion = q["x"]
        num_vars = q["num_vars"]
        prob_ppl = q["prob_perplexity"]

        orig_mregion = self.project_q(orig_mregion)

        negs, neg_idxs = self.sample_negatives(
            orig_mregion,
            orig_mregion.size(1),
            padding_count=padding_count,
        )

        q_aug = self.quantizer(orig_mregion_aug, produce_targets=False)
        orig_mregion_aug = q_aug["x"]
        orig_mregion_aug = self.project_q(orig_mregion_aug)

        negs_aug, neg_aug_idxs = self.sample_negatives(
            orig_mregion_aug,
            orig_mregion_aug.size(1),
            padding_count=padding_count,
        )

        with torch.no_grad():
            self.ema.model.eval()

            if self.cfg.ema_transformer_only:
                y, layer_results = self.ema.model.extract_features(
                    pre_encoder_features,
                    padding_mask=padding_mask,
                    min_layer=self.cfg.encoder_layers - self.average_top_k_layers,
                )
                y = {
                    "x": y,
                    "padding_mask": padding_mask,
                    "layer_results": layer_results,
                }
            else:
                y = self.ema.model.extract_features(
                    source=source,
                    padding_mask=orig_padding_mask,
                    mask=False,
                )

            target_layer_results = [l[2] for l in y["layer_results"]]

            permuted = False
            if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    tl.permute(1, 2, 0) for tl in target_layer_results  # TBC -> BCT
                ]
                permuted = True

            if self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    F.batch_norm(
                        tl.float(), running_mean=None, running_var=None, training=True
                    )
                    for tl in target_layer_results
                ]

            if self.cfg.instance_norm_target_layer:
                target_layer_results = [
                    F.instance_norm(tl.float()) for tl in target_layer_results
                ]

            if permuted:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
                ]

            if self.cfg.group_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-2:])
                    for tl in target_layer_results
                ]

            if self.cfg.layer_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-1:])
                    for tl in target_layer_results
                ]

            y = sum(target_layer_results) / len(target_layer_results)

            if self.cfg.layer_norm_targets:
                y = F.layer_norm(y.float(), y.shape[-1:])

            if self.cfg.instance_norm_targets:
                y = F.instance_norm(y.float().transpose(1, 2)).transpose(1, 2)

            if not permuted:
                y = y.transpose(0, 1)


        x_for_contr = x_aug[mask_indices].view(x_aug.size(0), -1, x_aug.size(-1))
        y_for_contr = y[mask_indices].view(y.size(0), -1, y.size(-1))
        x_aug = x_aug[mask_indices]
        y = y[mask_indices]
        x_aug = self.final_proj(x_aug)

        c_proj_x = self.contr_proj(x_for_contr)
        c_proj_y = self.contr_proj(y_for_contr.half())

        # Defining the k-means clustering module
        kmeans = KMeans(n_clusters=max(2, x.shape[1]//self.cfg.cluster_factor), mode='cosine', verbose=0)

        # normalize the sampled negatives of the original and the augmentation before clustering
        norm_orig_mregion = F.normalize(orig_mregion, dim=2)
        norm_orig_mregion_aug = F.normalize(orig_mregion_aug, dim=2)

        bsz, tsz, fsz = norm_orig_mregion.shape
        # pooling the normalized negatives before clustering
        data = torch.stack([norm_orig_mregion.view(-1, fsz),norm_orig_mregion_aug.view(-1, fsz)]).view(-1, fsz)
        data = data.to(torch.float)

        # clustering and obtaining the cluster IDs
        labels = kmeans.fit_predict(data)
        clus_labels, clus_labels_aug = torch.split(labels, bsz*tsz)
        clus_labels = torch.reshape(clus_labels, (bsz, tsz))
        neg_labels = clus_labels.view(-1)[neg_idxs]
        clus_labels = torch.reshape(clus_labels, (norm_orig_mregion.size(0), norm_orig_mregion.size(1), 1))
        neg_labels = neg_labels.view(norm_orig_mregion.size(0), norm_orig_mregion.size(1), self.n_negatives, 1).permute(
            2, 0, 1, 3
        )

        # filter those negatives which belong to the same cluster as the positive for the original sample
        filter_negs = (clus_labels == neg_labels).all(-1)

        clus_labels_aug = torch.reshape(clus_labels_aug, (bsz, tsz))
        neg_labels_aug = clus_labels_aug.view(-1)[neg_aug_idxs]
        clus_labels_aug = torch.reshape(clus_labels_aug, (norm_orig_mregion_aug.size(0), norm_orig_mregion_aug.size(1), 1))
        neg_labels_aug = neg_labels_aug.view(norm_orig_mregion_aug.size(0), norm_orig_mregion_aug.size(1), self.n_negatives, 1).permute(
            2, 0, 1, 3
        )

        # filter those negatives which belong to the same cluster as the positive for the original sample
        filter_negs_aug = (clus_labels_aug == neg_labels_aug).all(-1)

        # logits for cross-contrastive loss
        sims_s = self.compute_preds_scale(c_proj_x, orig_mregion, negs, filter_negs, self.cfg.scale_factor)
        logits_s = self.get_logits(sims_s)
        targets_s = self.get_targets(sims_s)

        # logits for cross-contrastive loss
        sims_t = self.compute_preds_scale(c_proj_y, orig_mregion_aug, negs_aug, filter_negs_aug, self.cfg.scale_factor)
        logits_t = self.get_logits(sims_t)
        targets_t = self.get_targets(sims_t)

        sz = x_aug.size(-1)

        if self.loss_beta == 0:
            loss_mse = F.mse_loss(x_aug.float(), y.float(), reduction="none").sum(dim=-1)
            loss_s = F.cross_entropy(logits_s, targets_s, reduction="sum")
            loss_t = F.cross_entropy(logits_t, targets_t, reduction="sum")
        else:
            loss = F.smooth_l1_loss(
                x_aug.float(), y.float(), reduction="none", beta=self.loss_beta
            ).sum(dim=-1)

        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(sz)
            scale_contrastive = 0.5

        result["prob_perplexity"] = prob_ppl
        result["features_pen"] = features_pen
        result["num_vars"] = num_vars

        result["losses"]["regression"] = (loss_mse.sum() * scale) + (loss_s * scale_contrastive) + (loss_t * scale_contrastive)

        result["contr_loss"] = loss_s.detach().item() + loss_t.detach().item()
        result["mse_loss"] = loss_mse.sum().detach().item()

        if "sample_size" not in result:
            result["sample_size"] = loss_mse.numel()
            result["contr_sample_size"] = targets_s.numel()

        with torch.no_grad():
            result["target_var"] = self.compute_var(y)
            result["pred_var"] = self.compute_var(x_aug.float())

        data.detach()

        if self.num_updates > 5000 and result["target_var"] < self.cfg.min_target_var:
            logger.error(
                f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
            )
            raise Exception(
                f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
            )
        if self.num_updates > 5000 and result["pred_var"] < self.cfg.min_pred_var:
            logger.error(
                f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
            )
            raise Exception(
                f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
            )

        if self.ema is not None:
            result["ema_decay"] = self.ema.get_decay() * 1000

        return result

    def get_extra_losses(self, net_output):
        pen = []

        if "prob_perplexity" in net_output:
            pen.append(
                (net_output["num_vars"] - net_output["prob_perplexity"])
                / net_output["num_vars"]
            )

        if "features_pen" in net_output:
            pen.append(net_output["features_pen"])

        return pen

    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y ** 2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs ** 2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()

    def extract_features(
        self, source, padding_mask, mask=False, layer=None
    ):
        res = self.forward(
            source,
            padding_mask,
            mask=mask,
            features_only=True,
            layer=layer,
        )
        return res

    def remove_pretraining_modules(self, last_layer=None):
        self.final_proj = None
        self.ema = None
        if last_layer is not None:
            self.encoder.layers = nn.ModuleList(
                l for i, l in enumerate(self.encoder.layers) if i <= last_layer
            )
