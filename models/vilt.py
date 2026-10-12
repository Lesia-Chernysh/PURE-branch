import torch
import torch.nn as nn
from lxt.explicit.models.vilt import ViltForQuestionAnswering, attnlrp as lrp_rule_composite

from transformers.activations import gelu, GELUActivation
################# LXT
import lxt.explicit.modules as lm
import lxt.explicit.functional as lf
import lxt.explicit.rules as rules
from lxt.explicit.core import Composite
from zennit.composites import LayerMapComposite
import zennit.rules as z_rules


class ViltCRPWrapper(nn.Module):
    """
    Give CondAttribution a simple positional interface while forwarding
    all arguments to ViLT by keyword.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def hf_model(self):
        return self.model

    def forward(
        self,
        pixel_values,
        inputs_embeds,
        attention_mask=None,
        token_type_ids=None,
        pixel_mask=None,
    ):
        outputs = self.model(
            pixel_values=pixel_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_mask=pixel_mask,
            return_dict=True,
        )

        return outputs.logits

class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)

def get_vilt(ckpt_path=None, n_class: int = None):
    return ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def get_vilt_for_crp(ckpt_path=None, n_class: int = None):
  """Use after registering LRP rules on vilt_vqa"""
  vilt_vqa = get_vilt(ckpt_path, n_class)
  return ViltCRPWrapper(vilt_vqa)


def get_vilt_canonizer():
    raise NotImplementedError


def get_vilt_composite():
  return Composite(
      {
          #nn.Linear: rules.EpsilonRule,
          nn.Tanh: rules.IdentityRule,
          GELUActivation: rules.IdentityRule,
          GeLU: rules.IdentityRule,
      },
      zennit_composite=LayerMapComposite([
          (nn.Conv2d, z_rules.Gamma(100)),
          (nn.Linear, z_rules.Gamma(0.05)),
      ])
  )
