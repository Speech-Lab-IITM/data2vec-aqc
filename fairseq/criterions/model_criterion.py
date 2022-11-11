# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import torch
import math

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


logger = logging.getLogger(__name__)


@dataclass
class ModelCriterionConfig(FairseqDataclass):
    loss_weights: Dict[str, float] = field(
        default_factory=dict,
        metadata={"help": "weights for the loss terms"},
    )
    log_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "additional output keys to log"},
    )


@register_criterion("model", dataclass=ModelCriterionConfig)
class ModelCriterion(FairseqCriterion):
    """
    This criterion relies on the model to supply losses.
    The losses should be a dictionary of name -> scalar returned by
    the model either by including it in the net_output dict or by
    implementing a get_losses(net_output, sample) method. The final loss is
    a scaled sum of all losses according to weights in loss_weights.
    If no weights are provided, then all losses are scaled by 1.0.

    The losses will be automatically logged. Additional keys from
    net_output dict can be logged via the log_keys parameter.
    """

    def __init__(self, task, loss_weights=None, log_keys=None):
        super().__init__(task)
        self.loss_weights = loss_weights
        self.log_keys = log_keys

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])

        scaled_losses = {}

        if hasattr(model, "get_losses"):
            losses = model.get_losses(net_output, sample)
        elif isinstance(net_output, dict) and "losses" in net_output:
            losses = net_output["losses"]
        else:
            raise Exception("Could not retrieve losses")

        for lk, p in losses.items():
            try:
                coef = 1.0 if len(self.loss_weights) == 0 else self.loss_weights[lk]
            except KeyError:
                logger.error(
                    f"weight for loss {lk} is not in loss_weights ({self.loss_weights})"
                )
                raise
            if coef != 0 and p is not None:
                scaled_losses[lk] = coef * p.float()

        loss = sum(scaled_losses.values())

        if "sample_size" in net_output:
            sample_size = net_output["sample_size"]
            contr_sample_size = net_output["contr_sample_size"]
        else:
            sample_size = loss.numel()

        if reduce and loss.numel() > 1:
            loss = loss.sum()

        extra_loss_weights = [0.1, 10]
        extra_losses = model.get_extra_losses(net_output)
        if torch.is_tensor(extra_losses):
            extra_losses = [extra_losses]
        if len(extra_loss_weights) == 1 and len(extra_losses) != 1:
            extra_loss_weights = [extra_loss_weights[0]] * len(extra_losses)
        assert len(extra_losses) == len(
            extra_loss_weights
        ), f"{len(extra_losses)}, {len(extra_loss_weights)}"
        for p, coef in zip(extra_losses, extra_loss_weights):
            if coef != 0 and p is not None:
                p = coef * p.float() * sample_size
                loss += p

        logging_output = {
            "loss": loss.data,
            "contr_loss": net_output["contr_loss"],
            "mse_loss": net_output["mse_loss"],
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            "contr_sample_size": contr_sample_size,
            "_world_size": 1,
        }

        for lk in self.log_keys:
            if lk in net_output and net_output[lk] is not None:
                if not torch.is_tensor(net_output[lk]) or net_output[lk].numel() == 1:
                    logging_output[lk] = float(net_output[lk])
                else:
                    for i, v in enumerate(net_output[lk]):
                        logging_output[f"{lk}_{i}"] = float(v)

        if len(scaled_losses) > 1:
            for lk, l in scaled_losses.items():
                if l.numel() > 1:
                    l = l.sum()
                logging_output[f"loss_{lk}"] = l.item()

        if "logs" in net_output:
            for lgw in net_output["logs"]:
                logging_output[lgw] = net_output["logs"][lgw]

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        contrastive_loss_sum = utils.item(sum(log.get("contr_loss", 0) for log in logging_outputs))
        mse_loss_sum = utils.item(sum(log.get("mse_loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        contr_sample_size = utils.item(
            sum(log.get("contr_sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar(
            "contrastive_loss", contrastive_loss_sum / (contr_sample_size or 1) / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "mse_loss", mse_loss_sum / (sample_size or 1) / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)

        builtin_keys = {
            "loss",
            "ntokens",
            "nsentences",
            "sample_size",
            "_world_size",
        }

        world_size = utils.item(
            sum(log.get("_world_size", 0) for log in logging_outputs)
        )

        for k in logging_outputs[0]:
            if k not in builtin_keys:
                val = sum(log.get(k, 0) for log in logging_outputs)
                if k.startswith("loss_"):
                    metrics.log_scalar(k, val / sample_size, sample_size, round=3)
                else:
                    metrics.log_scalar(k, val / world_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
