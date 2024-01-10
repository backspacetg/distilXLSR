import contextlib
import logging
from typing import Optional
from omegaconf import II, MISSING

import torch
import torch.nn as nn

from dataclasses import dataclass, field

from fairseq import utils
from fairseq.models import BaseFairseqModel, register_model
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask

from fairseq.models.distilXLSR.distilxlsr import DistilXLSR, DistilXLSRConfig

logger = logging.getLogger(__name__)
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])

@dataclass
class DistilXLSRCTCConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to wav2vec 2.0 model"}
    )
    # dropout
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout probability inside wav2vec 2.0 model"}
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside wav2vec 2.0 model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside wav2vec 2.0 model"
        },
    )
    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask (normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    mask_dropout: float = field(
        default=0.0,
        metadata={"help": "percent of masks to unmask for each sample"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    mask_channel_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune wav2vec for this many updates"}
    )
    feature_grad_mult: float = field(
        default=0.0, metadata={"help": "reset feature grad mult in wav2vec 2.0 to this"}
    )
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a layer in wav2vec 2.0"}
    )

    normalize: bool = II("task.normalize")
    data: str = II("task.data")

    blank_weight: float = 0
    blank_mode: str = "add"

@register_model("distilxlsr_ctc", dataclass=DistilXLSRCTCConfig)
class DistilXLSRCTC(BaseFairseqModel):
    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def __init__(self, cfg: DistilXLSRCTCConfig, model: DistilXLSR, input_size: int, output_size: int):
        super().__init__()
        self.cfg = cfg
        self.blank_weight = cfg.blank_weight
        self.blank_mode = cfg.blank_mode
        self.pretrained_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        self.proj = Linear(input_size, output_size)

    @classmethod
    def build_model(cls, cfg: DistilXLSRCTCConfig, task: FairseqTask):
        checkpoint = torch.load(cfg.w2v_path)
        pretrained_model_cfg = checkpoint["Config"]["model"]

        assert pretrained_model_cfg["normalize"] == cfg.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )
        for key in cfg.keys():
            pretrained_attribute = pretrained_model_cfg.get(key, None)
            if key[0] != "_" and pretrained_attribute is not None:
                if pretrained_model_cfg[key] != cfg.get(key):
                    logger.info("override {}: {} -> {}".format(
                        key, pretrained_model_cfg[key], cfg.get(key)
                    ))
                pretrained_model_cfg[key] = cfg.get(key)
            elif pretrained_attribute is None:
                logger.info(f"skip {key}")
        
        logger.info(pretrained_model_cfg)
        pretrained_model_cfg = DistilXLSRConfig(pretrained_model_cfg)
        model = DistilXLSR(pretrained_model_cfg)
        model.load_state_dict(checkpoint["Student"])

        return cls(cfg, model, pretrained_model_cfg.encoder_embed_dim, len(task.target_dictionary))

    # trainer传入的参数 告知模型目前更新的步数
    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates
    
    # beam search 解码需要的参数 主要目的是把encoder的输出重复beam_size次
    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out["padding_mask"].index_select(
                0, new_order
            )
        return encoder_out

    def max_positions(self):
        return None

    def get_logits(self, net_output, normalize=False):
        logits = net_output["encoder_out"]
        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., 0] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., 0] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            number_of_classes = logits.size(-1)
            masking_tensor = torch.ones(
                number_of_classes, device=logits.device
            ) * float("-inf")
            masking_tensor[0] = 0
            logits[net_output["padding_mask"].T] = masking_tensor.type_as(logits)

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits

    def get_normalized_probs(self, net_output, log_probs):
        logits = self.get_logits(net_output)

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self, source, padding_mask):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.cfg.apply_mask and self.training,
            "ret_layer_results": True,
        }


        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            (final_output, layer_results), padding_mask = self.pretrained_model.forward(**w2v_args)
            if self.pretrained_model.encoder.layer_norm_first:
                layer_hiddens = [i[2] for i in layer_results]
                layer_hiddens.pop(0)
                layer_hiddens.append(final_output)
            else:
                layer_hiddens = [i[0] for i in layer_results]
            x = layer_hiddens[-1]

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        x = self.final_dropout(x)
        x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "padding_mask": padding_mask,  # B x T,
            "layer_results": layer_results
        }

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

