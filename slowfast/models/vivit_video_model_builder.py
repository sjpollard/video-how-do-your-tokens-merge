from typing import Optional, Tuple

import torch
from torch import nn

import json

from transformers import VivitPreTrainedModel, VivitConfig, VivitModel
from .build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ViViT(VivitPreTrainedModel):
    def __init__(self, cfg):
        with open(cfg.VIVIT.CONFIG_PATH) as f:
            dict = json.load(f)
        config = VivitConfig(dict['video_size'][1], dict['video_size'][0], dict['tubelet_size'], dict['num_channels'], dict['hidden_size'], dict['num_hidden_layers'], dict['num_attention_heads'], dict['intermediate_size'], dict['hidden_act'], dict['hidden_dropout_prob'], dict['attention_probs_dropout_prob'], dict['initializer_range'], dict['layer_norm_eps'], dict['qkv_bias'], return_dict=False, attn_implementation='eager')
        super().__init__(config)

        self.num_labels = cfg.MODEL.NUM_CLASSES if cfg.EPICKITCHENS.NUM_CLASSES is None else cfg.EPICKITCHENS.NUM_CLASSES
        
        self.vivit = VivitModel(config, add_pooling_layer=False)

        if isinstance(self.num_labels, list):
            self.verb_classifier = nn.Linear(config.hidden_size, self.num_labels[0]) if self.num_labels[0] > 0 else nn.Identity()
            self.noun_classifier = nn.Linear(config.hidden_size, self.num_labels[1]) if self.num_labels[1] > 0 else nn.Identity()
        else:
            self.classifier = nn.Linear(config.hidden_size, self.num_labels) if self.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        pixel_values = pixel_values[0].permute(0, 2, 1, 3, 4)

        outputs = self.vivit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        if isinstance(self.num_labels, list):
            verb_logits = self.verb_classifier(sequence_output[:, 0, :])
            noun_logits = self.noun_classifier(sequence_output[:, 0, :])
            logits = (verb_logits, noun_logits)
        else:
            logits = self.classifier(sequence_output[:, 0, :])

        return logits