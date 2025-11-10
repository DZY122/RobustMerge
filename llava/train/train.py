# # Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# # Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# #    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
# #
# #    Licensed under the Apache License, Version 2.0 (the "License");
# #    you may not use this file except in compliance with the License.
# #    You may obtain a copy of the License at
# #
# #        http://www.apache.org/licenses/LICENSE-2.0
# #
# #    Unless required by applicable law or agreed to in writing, software
# #    distributed under the License is distributed on an "AS IS" BASIS,
# #    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# #    See the License for the specific language governing permissions and
# #    limitations under the License.

# import os
# import copy
# from dataclasses import dataclass, field
# import json
# import logging
# import pathlib
# from typing import Dict, Optional, Sequence, List

# import torch

# import transformers
# import tokenizers

# from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from torch.utils.data import Dataset
# from llava.train.llava_trainer import LLaVATrainer

# from llava import conversation as conversation_lib
# from llava.model import *
# from llava.model.svd_tuning import (
#     LinearSVDAdapter,
#     SVDLinearConfig,
#     apply_svd_tuning,
#     load_svd_adapters,
#     load_svd_config,
# )
# from llava.mm_utils import tokenizer_image_token

# from PIL import Image


# local_rank = None


# def rank0_print(*args):
#     if local_rank == 0:
#         print(*args)


# from packaging import version
# IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


# @dataclass
# class ModelArguments:
#     model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
#     version: Optional[str] = field(default="v0")
#     freeze_backbone: bool = field(default=False)
#     tune_mm_mlp_adapter: bool = field(default=False)
#     vision_tower: Optional[str] = field(default=None)
#     mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
#     pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
#     mm_projector_type: Optional[str] = field(default='linear')
#     mm_use_im_start_end: bool = field(default=False)
#     mm_use_im_patch_token: bool = field(default=True)
#     mm_patch_merge_type: Optional[str] = field(default='flat')
#     mm_vision_select_feature: Optional[str] = field(default="patch")


# @dataclass
# class DataArguments:
#     data_path: str = field(default=None,
#                            metadata={"help": "Path to the training data."})
#     lazy_preprocess: bool = False
#     is_multimodal: bool = False
#     image_folder: Optional[str] = field(default=None)
#     image_aspect_ratio: str = 'square'


# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     cache_dir: Optional[str] = field(default=None)
#     optim: str = field(default="adamw_torch")
#     remove_unused_columns: bool = field(default=False)
#     freeze_mm_mlp_adapter: bool = field(default=False)
#     mpt_attn_impl: Optional[str] = field(default="triton")
#     model_max_length: int = field(
#         default=512,
#         metadata={
#             "help":
#             "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
#         },
#     )
#     double_quant: bool = field(
#         default=True,
#         metadata={"help": "Compress the quantization statistics through double quantization."}
#     )
#     quant_type: str = field(
#         default="nf4",
#         metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
#     )
#     bits: int = field(
#         default=16,
#         metadata={"help": "How many bits to use."}
#     )
#     svd_enable: bool = False
#     svd_weight_path: str = ""
#     svd_num_groups: int = 4
#     svd_selected_group: int = 1
#     mm_projector_lr: Optional[float] = None
#     group_by_modality_length: bool = field(default=False)


# def maybe_zero_3(param, ignore_status=False, name=None):
#     from deepspeed import zero
#     from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
#     if hasattr(param, "ds_id"):
#         if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
#             if not ignore_status:
#                 logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
#         with zero.GatheredParameters([param]):
#             param = param.data.detach().cpu().clone()
#     else:
#         param = param.detach().cpu().clone()
#     return param


# def _collect_state_dict(named_params, predicate, require_grad_only=True):
#     params = [(k, t) for k, t in named_params if predicate(k, t)]
#     if require_grad_only:
#         params = [(k, t) for k, t in params if t.requires_grad]
#     if not params:
#         return {}

#     zero_available = False
#     try:
#         from deepspeed import zero  # type: ignore
#         zero_available = True
#     except Exception:
#         zero_available = False

#     if zero_available and any(hasattr(param, "ds_id") for _, param in params):
#         gathered = {}
#         param_list = [param for _, param in params]
#         with zero.GatheredParameters(param_list, modifier_rank=0):
#             for name, param in params:
#                 gathered[name] = param.detach().cpu().clone()
#         return gathered

#     return {name: param.detach().cpu().clone() for name, param in params}


# def get_svd_state_dict(named_params, require_grad_only=True):
#     return _collect_state_dict(named_params, lambda name, _: "svd_" in name, require_grad_only)


# def get_non_svd_state_dict(named_params, require_grad_only=True):
#     return _collect_state_dict(named_params, lambda name, _: "svd_" not in name, require_grad_only)


# def _resolve_weight_file(base_path: str, filename: str) -> Optional[str]:
#     if os.path.isdir(base_path):
#         candidate = os.path.join(base_path, filename)
#         if os.path.exists(candidate):
#             return candidate
#     if os.path.isfile(base_path) and os.path.basename(base_path) == filename:
#         return base_path

#     try:
#         from huggingface_hub import hf_hub_download  # type: ignore
#     except Exception:
#         return None

#     try:
#         return hf_hub_download(repo_id=base_path, filename=filename)
#     except Exception:
#         return None


# def _load_additional_trainables(model: torch.nn.Module, weight_path: str):
#     non_svd_file = _resolve_weight_file(weight_path, 'non_lora_trainables.bin')
#     if non_svd_file is None:
#         rank0_print(f"No non-SVD trainables found at {weight_path}")
#         return

#     state = torch.load(non_svd_file, map_location='cpu')
#     state = {(k[11:] if k.startswith('base_model.') else k): v for k, v in state.items()}
#     if any(k.startswith('model.model.') for k in state):
#         state = {(k[6:] if k.startswith('model.') else k): v for k, v in state.items()}
#     missing, unexpected = model.load_state_dict(state, strict=False)
#     if missing:
#         rank0_print(f"Warning: missing keys when loading non-SVD parameters: {missing[:5]}{'...' if len(missing) > 5 else ''}")
#     if unexpected:
#         rank0_print(f"Warning: unexpected keys when loading non-SVD parameters: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")


# def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
#     to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
#     to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
#     return to_return


# def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
#                                    output_dir: str):
#     """Collects the state dict and dump to disk."""

#     if getattr(trainer.args, "tune_mm_mlp_adapter", False):
#         # Only save Adapter
#         keys_to_match = ['mm_projector']
#         if getattr(trainer.args, "use_im_start_end", False):
#             keys_to_match.extend(['embed_tokens', 'embed_in'])

#         weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
#         trainer.model.config.save_pretrained(output_dir)

#         current_folder = output_dir.split('/')[-1]
#         parent_folder = os.path.dirname(output_dir)
#         if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
#             if current_folder.startswith('checkpoint-'):
#                 mm_projector_folder = os.path.join(parent_folder, "mm_projector")
#                 os.makedirs(mm_projector_folder, exist_ok=True)
#                 torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
#             else:
#                 torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
#         return

#     if trainer.deepspeed:
#         torch.cuda.synchronize()
#         trainer.save_model(output_dir)
#         return

#     state_dict = trainer.model.state_dict()
#     if trainer.args.should_save:
#         cpu_state_dict = {
#             key: value.cpu()
#             for key, value in state_dict.items()
#         }
#         del state_dict
#         trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


# def smart_tokenizer_and_embedding_resize(
#     special_tokens_dict: Dict,
#     tokenizer: transformers.PreTrainedTokenizer,
#     model: transformers.PreTrainedModel,
# ):
#     """Resize tokenizer and embedding.

#     Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
#     """
#     num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
#     model.resize_token_embeddings(len(tokenizer))

#     if num_new_tokens > 0:
#         input_embeddings = model.get_input_embeddings().weight.data
#         output_embeddings = model.get_output_embeddings().weight.data

#         input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
#             dim=0, keepdim=True)
#         output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
#             dim=0, keepdim=True)

#         input_embeddings[-num_new_tokens:] = input_embeddings_avg
#         output_embeddings[-num_new_tokens:] = output_embeddings_avg


# def _tokenize_fn(strings: Sequence[str],
#                  tokenizer: transformers.PreTrainedTokenizer) -> Dict:
#     """Tokenize a list of strings."""
#     tokenized_list = [
#         tokenizer(
#             text,
#             return_tensors="pt",
#             padding="longest",
#             max_length=tokenizer.model_max_length,
#             truncation=True,
#         ) for text in strings
#     ]
#     input_ids = labels = [
#         tokenized.input_ids[0] for tokenized in tokenized_list
#     ]
#     input_ids_lens = labels_lens = [
#         tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
#         for tokenized in tokenized_list
#     ]
#     return dict(
#         input_ids=input_ids,
#         labels=labels,
#         input_ids_lens=input_ids_lens,
#         labels_lens=labels_lens,
#     )


# def _mask_targets(target, tokenized_lens, speakers):
#     # cur_idx = 0
#     cur_idx = tokenized_lens[0]
#     tokenized_lens = tokenized_lens[1:]
#     target[:cur_idx] = IGNORE_INDEX
#     for tokenized_len, speaker in zip(tokenized_lens, speakers):
#         if speaker == "human":
#             target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
#         cur_idx += tokenized_len


# def _add_speaker_and_signal(header, source, get_conversation=True):
#     """Add speaker and start/end signal on each round."""
#     BEGIN_SIGNAL = "### "
#     END_SIGNAL = "\n"
#     conversation = header
#     for sentence in source:
#         from_str = sentence["from"]
#         if from_str.lower() == "human":
#             from_str = conversation_lib.default_conversation.roles[0]
#         elif from_str.lower() == "gpt":
#             from_str = conversation_lib.default_conversation.roles[1]
#         else:
#             from_str = 'unknown'
#         sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
#                              sentence["value"] + END_SIGNAL)
#         if get_conversation:
#             conversation += sentence["value"]
#     conversation += BEGIN_SIGNAL
#     return conversation


# def preprocess_multimodal(
#     sources: Sequence[str],
#     data_args: DataArguments
# ) -> Dict:
#     is_multimodal = data_args.is_multimodal
#     if not is_multimodal:
#         return sources

#     for source in sources:
#         for sentence in source:
#             if DEFAULT_IMAGE_TOKEN in sentence['value']:
#                 sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
#                 sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
#                 sentence['value'] = sentence['value'].strip()
#                 if "mmtag" in conversation_lib.default_conversation.version:
#                     sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
#             replace_token = DEFAULT_IMAGE_TOKEN
#             if data_args.mm_use_im_start_end:
#                 replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
#             sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

#     return sources


# def preprocess_llama_2(
#     sources,
#     tokenizer: transformers.PreTrainedTokenizer,
#     has_image: bool = False
# ) -> Dict:
#     conv = conversation_lib.default_conversation.copy()
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

#     # Apply prompt templates
#     conversations = []
#     for i, source in enumerate(sources):
#         if roles[source[0]["from"]] != conv.roles[0]:
#             # Skip the first one if it is not from human
#             source = source[1:]

#         conv.messages = []
#         for j, sentence in enumerate(source):
#             role = roles[sentence["from"]]
#             assert role == conv.roles[j % 2], f"{i}"
#             conv.append_message(role, sentence["value"])
#         conversations.append(conv.get_prompt())

#     # Tokenize conversations

#     if has_image:
#         input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
#     else:
#         input_ids = tokenizer(
#             conversations,
#             return_tensors="pt",
#             padding="longest",
#             max_length=tokenizer.model_max_length,
#             truncation=True,
#         ).input_ids

#     targets = input_ids.clone()

#     assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

#     # Mask targets
#     sep = "[/INST] "
#     for conversation, target in zip(conversations, targets):
#         total_len = int(target.ne(tokenizer.pad_token_id).sum())

#         rounds = conversation.split(conv.sep2)
#         cur_len = 1
#         target[:cur_len] = IGNORE_INDEX
#         for i, rou in enumerate(rounds):
#             if rou == "":
#                 break

#             parts = rou.split(sep)
#             if len(parts) != 2:
#                 break
#             parts[0] += sep

#             if has_image:
#                 round_len = len(tokenizer_image_token(rou, tokenizer))
#                 instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
#             else:
#                 round_len = len(tokenizer(rou).input_ids)
#                 instruction_len = len(tokenizer(parts[0]).input_ids) - 2

#             target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

#             cur_len += round_len
#         target[cur_len:] = IGNORE_INDEX

#         if cur_len < tokenizer.model_max_length:
#             if cur_len != total_len:
#                 target[:] = IGNORE_INDEX
#                 print(
#                     f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
#                     f" (ignored)"
#                 )

#     return dict(
#         input_ids=input_ids,
#         labels=targets,
#     )


# def preprocess_v1(
#     sources,
#     tokenizer: transformers.PreTrainedTokenizer,
#     has_image: bool = False
# ) -> Dict:
#     conv = conversation_lib.default_conversation.copy()
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

#     # Apply prompt templates
#     conversations = []
#     for i, source in enumerate(sources):
#         if roles[source[0]["from"]] != conv.roles[0]:
#             # Skip the first one if it is not from human
#             source = source[1:]

#         conv.messages = []
#         for j, sentence in enumerate(source):
#             role = roles[sentence["from"]]
#             assert role == conv.roles[j % 2], f"{i}"
#             conv.append_message(role, sentence["value"])
#         conversations.append(conv.get_prompt())

#     # Tokenize conversations

#     if has_image:
#         input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
#     else:
#         input_ids = tokenizer(
#             conversations,
#             return_tensors="pt",
#             padding="longest",
#             max_length=tokenizer.model_max_length,
#             truncation=True,
#         ).input_ids

#     targets = input_ids.clone()

#     assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

#     # Mask targets
#     sep = conv.sep + conv.roles[1] + ": "
#     for conversation, target in zip(conversations, targets):
#         total_len = int(target.ne(tokenizer.pad_token_id).sum())

#         rounds = conversation.split(conv.sep2)
#         cur_len = 1
#         target[:cur_len] = IGNORE_INDEX
#         for i, rou in enumerate(rounds):
#             if rou == "":
#                 break

#             parts = rou.split(sep)
#             if len(parts) != 2:
#                 break
#             parts[0] += sep

#             if has_image:
#                 round_len = len(tokenizer_image_token(rou, tokenizer))
#                 instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
#             else:
#                 round_len = len(tokenizer(rou).input_ids)
#                 instruction_len = len(tokenizer(parts[0]).input_ids) - 2

#             if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
#                 round_len -= 1
#                 instruction_len -= 1

#             target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

#             cur_len += round_len
#         target[cur_len:] = IGNORE_INDEX

#         if cur_len < tokenizer.model_max_length:
#             if cur_len != total_len:
#                 target[:] = IGNORE_INDEX
#                 print(
#                     f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
#                     f" (ignored)"
#                 )

#     return dict(
#         input_ids=input_ids,
#         labels=targets,
#     )


# def preprocess_mpt(
#     sources,
#     tokenizer: transformers.PreTrainedTokenizer,
#     has_image: bool = False
# ) -> Dict:
#     conv = conversation_lib.default_conversation.copy()
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

#     # Apply prompt templates
#     conversations = []
#     for i, source in enumerate(sources):
#         if roles[source[0]["from"]] != conv.roles[0]:
#             # Skip the first one if it is not from human
#             source = source[1:]

#         conv.messages = []
#         for j, sentence in enumerate(source):
#             role = roles[sentence["from"]]
#             assert role == conv.roles[j % 2], f"{i}"
#             conv.append_message(role, sentence["value"])
#         conversations.append(conv.get_prompt())

#     # Tokenize conversations

#     if has_image:
#         input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
#     else:
#         input_ids = tokenizer(
#             conversations,
#             return_tensors="pt",
#             padding="longest",
#             max_length=tokenizer.model_max_length,
#             truncation=True,
#         ).input_ids

#     targets = input_ids.clone()
#     assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

#     # Mask targets
#     sep = conv.sep + conv.roles[1]
#     for conversation, target in zip(conversations, targets):
#         total_len = int(target.ne(tokenizer.pad_token_id).sum())

#         rounds = conversation.split(conv.sep)
#         re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
#         for conv_idx in range(3, len(rounds), 2):
#             re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
#         cur_len = 0
#         target[:cur_len] = IGNORE_INDEX
#         for i, rou in enumerate(re_rounds):
#             if rou == "":
#                 break

#             parts = rou.split(sep)
#             if len(parts) != 2:
#                 break
#             parts[0] += sep

#             if has_image:
#                 round_len = len(tokenizer_image_token(rou, tokenizer))
#                 instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
#             else:
#                 round_len = len(tokenizer(rou).input_ids)
#                 instruction_len = len(tokenizer(parts[0]).input_ids) - 1

#             if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
#                 round_len += 1
#                 instruction_len += 1

#             target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

#             cur_len += round_len
#         target[cur_len:] = IGNORE_INDEX

#         if cur_len < tokenizer.model_max_length:
#             if cur_len != total_len:
#                 target[:] = IGNORE_INDEX
#                 print(
#                     f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
#                     f" (ignored)"
#                 )

#     return dict(
#         input_ids=input_ids,
#         labels=targets,
#     )


# def preprocess_plain(
#     sources: Sequence[str],
#     tokenizer: transformers.PreTrainedTokenizer,
# ) -> Dict:
#     # add end signal and concatenate together
#     conversations = []
#     for source in sources:
#         assert len(source) == 2
#         assert DEFAULT_IMAGE_TOKEN in source[0]['value']
#         source[0]['value'] = DEFAULT_IMAGE_TOKEN
#         conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
#         conversations.append(conversation)
#     # tokenize conversations
#     input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
#     targets = copy.deepcopy(input_ids)
#     for target, source in zip(targets, sources):
#         tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
#         target[:tokenized_len] = IGNORE_INDEX

#     return dict(input_ids=input_ids, labels=targets)


# def preprocess(
#     sources: Sequence[str],
#     tokenizer: transformers.PreTrainedTokenizer,
#     has_image: bool = False
# ) -> Dict:
#     """
#     Given a list of sources, each is a conversation list. This transform:
#     1. Add signal '### ' at the beginning each sentence, with end signal '\n';
#     2. Concatenate conversations together;
#     3. Tokenize the concatenated conversation;
#     4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
#     """
#     if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
#         return preprocess_plain(sources, tokenizer)
#     if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
#         return preprocess_llama_2(sources, tokenizer, has_image=has_image)
#     if conversation_lib.default_conversation.version.startswith("v1"):
#         return preprocess_v1(sources, tokenizer, has_image=has_image)
#     if conversation_lib.default_conversation.version == "mpt":
#         return preprocess_mpt(sources, tokenizer, has_image=has_image)
#     # add end signal and concatenate together
#     conversations = []
#     for source in sources:
#         header = f"{conversation_lib.default_conversation.system}\n\n"
#         conversation = _add_speaker_and_signal(header, source)
#         conversations.append(conversation)
#     # tokenize conversations
#     def get_tokenize_len(prompts):
#         return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

#     if has_image:
#         input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
#     else:
#         conversations_tokenized = _tokenize_fn(conversations, tokenizer)
#         input_ids = conversations_tokenized["input_ids"]

#     targets = copy.deepcopy(input_ids)
#     for target, source in zip(targets, sources):
#         if has_image:
#             tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
#         else:
#             tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
#         speakers = [sentence["from"] for sentence in source]
#         _mask_targets(target, tokenized_lens, speakers)

#     return dict(input_ids=input_ids, labels=targets)


# class LazySupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""

#     def __init__(self, data_path: str,
#                  tokenizer: transformers.PreTrainedTokenizer,
#                  data_args: DataArguments):
#         super(LazySupervisedDataset, self).__init__()
#         list_data_dict = json.load(open(data_path, "r"))

#         rank0_print("Formatting inputs...Skip in lazy mode")
#         self.tokenizer = tokenizer
#         self.list_data_dict = list_data_dict
#         self.data_args = data_args

#     def __len__(self):
#         return len(self.list_data_dict)

#     @property
#     def lengths(self):
#         length_list = []
#         for sample in self.list_data_dict:
#             img_tokens = 128 if 'image' in sample else 0
#             length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
#         return length_list

#     @property
#     def modality_lengths(self):
#         length_list = []
#         for sample in self.list_data_dict:
#             cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
#             cur_len = cur_len if 'image' in sample else -cur_len
#             length_list.append(cur_len)
#         return length_list

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         sources = self.list_data_dict[i]
#         if isinstance(i, int):
#             sources = [sources]
#         assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
#         if 'image' in sources[0]:
#             image_file = self.list_data_dict[i]['image']
#             image_folder = self.data_args.image_folder
#             processor = self.data_args.image_processor
#             image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
#             if self.data_args.image_aspect_ratio == 'pad':
#                 def expand2square(pil_img, background_color):
#                     width, height = pil_img.size
#                     if width == height:
#                         return pil_img
#                     elif width > height:
#                         result = Image.new(pil_img.mode, (width, width), background_color)
#                         result.paste(pil_img, (0, (width - height) // 2))
#                         return result
#                     else:
#                         result = Image.new(pil_img.mode, (height, height), background_color)
#                         result.paste(pil_img, ((height - width) // 2, 0))
#                         return result
#                 image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
#                 image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
#             else:
#                 image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
#             sources = preprocess_multimodal(
#                 copy.deepcopy([e["conversations"] for e in sources]),
#                 self.data_args)
#         else:
#             sources = copy.deepcopy([e["conversations"] for e in sources])
#         data_dict = preprocess(
#             sources,
#             self.tokenizer,
#             has_image=('image' in self.list_data_dict[i]))
#         if isinstance(i, int):
#             data_dict = dict(input_ids=data_dict["input_ids"][0],
#                              labels=data_dict["labels"][0])

#         # image exist in the data
#         if 'image' in self.list_data_dict[i]:
#             data_dict['image'] = image
#         elif self.data_args.is_multimodal:
#             # image does not exist in the data, but the model is multimodal
#             crop_size = self.data_args.image_processor.crop_size
#             data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
#         return data_dict


# @dataclass
# class DataCollatorForSupervisedDataset(object):
#     """Collate examples for supervised fine-tuning."""

#     tokenizer: transformers.PreTrainedTokenizer

#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         input_ids, labels = tuple([instance[key] for instance in instances]
#                                   for key in ("input_ids", "labels"))
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids,
#             batch_first=True,
#             padding_value=self.tokenizer.pad_token_id)
#         labels = torch.nn.utils.rnn.pad_sequence(labels,
#                                                  batch_first=True,
#                                                  padding_value=IGNORE_INDEX)
#         input_ids = input_ids[:, :self.tokenizer.model_max_length]
#         labels = labels[:, :self.tokenizer.model_max_length]
#         batch = dict(
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#         )

#         if 'image' in instances[0]:
#             images = [instance['image'] for instance in instances]
#             if all(x is not None and x.shape == images[0].shape for x in images):
#                 batch['images'] = torch.stack(images)
#             else:
#                 batch['images'] = images

#         return batch


# def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
#                                 data_args) -> Dict:
#     """Make dataset and collator for supervised fine-tuning."""
#     train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
#                                 data_path=data_args.data_path,
#                                 data_args=data_args)
#     data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
#     return dict(train_dataset=train_dataset,
#                 eval_dataset=None,
#                 data_collator=data_collator)


# def train(attn_implementation=None):
#     global local_rank

#     parser = transformers.HfArgumentParser(
#         (ModelArguments, DataArguments, TrainingArguments))
#     model_args, data_args, training_args = parser.parse_args_into_dataclasses()
#     local_rank = training_args.local_rank
#     compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

#     bnb_model_from_pretrained_args = {}
#     if training_args.bits in [4, 8]:
#         from transformers import BitsAndBytesConfig
#         bnb_model_from_pretrained_args.update(dict(
#             device_map={"": training_args.device},
#             load_in_4bit=training_args.bits == 4,
#             load_in_8bit=training_args.bits == 8,
#             quantization_config=BitsAndBytesConfig(
#                 load_in_4bit=training_args.bits == 4,
#                 load_in_8bit=training_args.bits == 8,
#                 llm_int8_skip_modules=["mm_projector"],
#                 llm_int8_threshold=6.0,
#                 llm_int8_has_fp16_weight=False,
#                 bnb_4bit_compute_dtype=compute_dtype,
#                 bnb_4bit_use_double_quant=training_args.double_quant,
#                 bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
#             )
#         ))

#     if model_args.vision_tower is not None:
#         if 'mpt' in model_args.model_name_or_path:
#             config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
#             config.attn_config['attn_impl'] = training_args.mpt_attn_impl
#             model = LlavaMptForCausalLM.from_pretrained(
#                 model_args.model_name_or_path,
#                 config=config,
#                 cache_dir=training_args.cache_dir,
#                 **bnb_model_from_pretrained_args
#             )
#         else:
#             model = LlavaLlamaForCausalLM.from_pretrained(
#                 model_args.model_name_or_path,
#                 cache_dir=training_args.cache_dir,
#                 attn_implementation=attn_implementation,
#                 torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
#                 **bnb_model_from_pretrained_args
#             )
#     else:
#         model = transformers.LlamaForCausalLM.from_pretrained(
#             model_args.model_name_or_path,
#             cache_dir=training_args.cache_dir,
#             attn_implementation=attn_implementation,
#             torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
#             **bnb_model_from_pretrained_args
#         )
#     model.config.use_cache = False

#     if model_args.freeze_backbone:
#         model.model.requires_grad_(False)

#     if training_args.bits in [4, 8]:
#         from peft import prepare_model_for_kbit_training
#         model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
#         model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

#     if training_args.gradient_checkpointing:
#         if hasattr(model, "enable_input_require_grads"):
#             model.enable_input_require_grads()
#         else:
#             def make_inputs_require_grad(module, input, output):
#                 output.requires_grad_(True)
#             model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

#     if training_args.svd_enable:
#         if training_args.bits == 16:
#             if training_args.bf16:
#                 model.to(torch.bfloat16)
#             if training_args.fp16:
#                 model.to(torch.float16)
#         rank0_print("Applying SVD-based adapters...")
#         svd_config = SVDLinearConfig(
#             num_groups=training_args.svd_num_groups,
#             selected_group=training_args.svd_selected_group,
#         )
#         apply_svd_tuning(model, svd_config)
#         inferred_dim = None
#         for module in model.modules():
#             if isinstance(module, LinearSVDAdapter):
#                 inferred_dim = module.adapter_dim
#                 break
#         if inferred_dim is not None:
#             rank0_print(
#                 f"SVD linear adapter dimension inferred from selected group: {inferred_dim}"
#                 f" (≈ min(in_features, out_features) / {training_args.svd_num_groups})"
#             )
#         model.config.svd_tuning = svd_config.to_dict()

#         config_source = training_args.svd_weight_path
#         if config_source:
#             weight_locator = config_source if not os.path.isfile(config_source) else os.path.dirname(config_source)
#             try:
#                 resolved_config = config_source
#                 loaded_config = load_svd_config(config_source)
#             except (OSError, json.JSONDecodeError):
#                 resolved_config = _resolve_weight_file(weight_locator, 'svd_config.json')
#                 if resolved_config is None:
#                     raise FileNotFoundError(f"Unable to locate svd_config.json under {config_source}")
#                 loaded_config = load_svd_config(resolved_config)
#             if loaded_config.to_dict() != svd_config.to_dict():
#                 rank0_print("Warning: Loaded SVD config does not match training config. Using training config.")
#             adapter_source = config_source
#             if not (os.path.isdir(adapter_source) and os.path.exists(os.path.join(adapter_source, 'adapter_model.bin'))):
#                 resolved_adapter = _resolve_weight_file(weight_locator, 'adapter_model.bin')
#                 if resolved_adapter is not None:
#                     adapter_source = resolved_adapter
#             if not (os.path.isdir(adapter_source) or os.path.isfile(adapter_source)):
#                 raise FileNotFoundError(f"Unable to locate adapter_model.bin under {weight_locator}")
#             load_svd_adapters(model, adapter_source)
#             _load_additional_trainables(model, weight_locator)

#     if 'mpt' in model_args.model_name_or_path:
#         tokenizer = transformers.AutoTokenizer.from_pretrained(
#             model_args.model_name_or_path,
#             cache_dir=training_args.cache_dir,
#             model_max_length=training_args.model_max_length,
#             padding_side="right"
#         )
#     else:
#         tokenizer = transformers.AutoTokenizer.from_pretrained(
#             model_args.model_name_or_path,
#             cache_dir=training_args.cache_dir,
#             model_max_length=training_args.model_max_length,
#             padding_side="right",
#             use_fast=False,
#         )

#     if model_args.version == "v0":
#         if tokenizer.pad_token is None:
#             smart_tokenizer_and_embedding_resize(
#                 special_tokens_dict=dict(pad_token="[PAD]"),
#                 tokenizer=tokenizer,
#                 model=model,
#             )
#     elif model_args.version == "v0.5":
#         tokenizer.pad_token = tokenizer.unk_token
#     else:
#         tokenizer.pad_token = tokenizer.unk_token
#         if model_args.version in conversation_lib.conv_templates:
#             conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
#         else:
#             conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

#     if model_args.vision_tower is not None:
#         model.get_model().initialize_vision_modules(
#             model_args=model_args,
#             fsdp=training_args.fsdp
#         )
        
#         vision_tower = model.get_vision_tower()
#         vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

#         data_args.image_processor = vision_tower.image_processor
#         data_args.is_multimodal = True

#         model.config.image_aspect_ratio = data_args.image_aspect_ratio
#         model.config.tokenizer_padding_side = tokenizer.padding_side
#         model.config.tokenizer_model_max_length = tokenizer.model_max_length

#         model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
#         if model_args.tune_mm_mlp_adapter:
#             model.requires_grad_(False)
#             for p in model.get_model().mm_projector.parameters():
#                 p.requires_grad = True

#         model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
#         if training_args.freeze_mm_mlp_adapter:
#             for p in model.get_model().mm_projector.parameters():
#                 p.requires_grad = False

#         if training_args.bits in [4, 8]:
#             model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

#         model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
#         model.config.mm_projector_lr = training_args.mm_projector_lr
#         training_args.use_im_start_end = model_args.mm_use_im_start_end
#         model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
#         model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

#     if training_args.svd_enable:
#         extra_trainable = []
#         if model_args.tune_mm_mlp_adapter or training_args.mm_projector_lr is not None:
#             extra_trainable.append('mm_projector')
#         for name, param in model.named_parameters():
#             if "svd_" in name or any(token in name for token in extra_trainable):
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False

#     data_module = make_supervised_data_module(tokenizer=tokenizer,
#                                               data_args=data_args)
#     trainer = LLaVATrainer(model=model,
#                     tokenizer=tokenizer,
#                     args=training_args,
#                     **data_module)

#     if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
#         trainer.train(resume_from_checkpoint=True)
#     else:
#         trainer.train()
#     trainer.save_state()

#     model.config.use_cache = True

#     if training_args.svd_enable:
#         if training_args.local_rank == 0 or training_args.local_rank == -1:
#             model.config.save_pretrained(training_args.output_dir)
#             svd_config = SVDLinearConfig.from_dict(model.config.svd_tuning)
#             os.makedirs(training_args.output_dir, exist_ok=True)
#             svd_state = get_svd_state_dict(model.named_parameters())
#             torch.save(svd_state, os.path.join(training_args.output_dir, 'adapter_model.bin'))
#             with open(os.path.join(training_args.output_dir, 'svd_config.json'), 'w') as f:
#                 json.dump(svd_config.to_dict(), f)
#             non_svd_state = get_non_svd_state_dict(model.named_parameters())
#             torch.save(non_svd_state, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
#     else:
#         safe_save_model_for_hf_trainer(trainer=trainer,
#                                        output_dir=training_args.output_dir)


# if __name__ == "__main__":
#     train()


# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.model.svd_tuning import (
    LinearSVDAdapter,
    SVDLinearConfig,
    apply_svd_tuning,
    load_svd_adapters,
    load_svd_config,
)
from llava.mm_utils import tokenizer_image_token

from PIL import Image


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    svd_enable: bool = False
    svd_weight_path: str = ""
    svd_num_groups: int = 4
    svd_selected_group: int = 1
    svd_match_lora_rank: Optional[int] = None
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def _collect_state_dict(named_params, predicate, require_grad_only=True):
    params = [(k, t) for k, t in named_params if predicate(k, t)]
    if require_grad_only:
        params = [(k, t) for k, t in params if t.requires_grad]
    if not params:
        return {}

    zero_available = False
    try:
        from deepspeed import zero  # type: ignore
        zero_available = True
    except Exception:
        zero_available = False

    if zero_available:
        ds_params = [(name, param) for name, param in params if hasattr(param, "ds_id")]
        non_ds_params = [(name, param) for name, param in params if not hasattr(param, "ds_id")]
        if ds_params:
            gathered = {}

            dist = getattr(torch, "distributed", None)
            should_store = True
            if dist is not None and dist.is_available() and dist.is_initialized():
                should_store = dist.get_rank() == 0

            def _gather_chunk(chunk):
                if not chunk:
                    return
                with zero.GatheredParameters([p for _, p in chunk], modifier_rank=0):
                    if should_store:
                        for name, param in chunk:
                            gathered[name] = param.detach().cpu().clone()

            max_chunk_bytes = 256 * 1024 * 1024  # 256MB
            current_chunk = []
            current_size = 0
            for name, param in ds_params:
                current_chunk.append((name, param))
                current_size += param.numel() * param.element_size()
                if current_size >= max_chunk_bytes:
                    _gather_chunk(current_chunk)
                    current_chunk = []
                    current_size = 0
            _gather_chunk(current_chunk)

            if should_store:
                for name, param in non_ds_params:
                    gathered[name] = param.detach().cpu().clone()
                return gathered
            return {}

    return {name: param.detach().cpu().clone() for name, param in params}


def get_svd_state_dict(named_params, require_grad_only=True):
    return _collect_state_dict(named_params, lambda name, _: "svd_" in name, require_grad_only)


def get_non_svd_state_dict(named_params, require_grad_only=True):
    return _collect_state_dict(named_params, lambda name, _: "svd_" not in name, require_grad_only)


def _resolve_weight_file(base_path: str, filename: str) -> Optional[str]:
    if os.path.isdir(base_path):
        candidate = os.path.join(base_path, filename)
        if os.path.exists(candidate):
            return candidate
    if os.path.isfile(base_path) and os.path.basename(base_path) == filename:
        return base_path

    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception:
        return None

    try:
        return hf_hub_download(repo_id=base_path, filename=filename)
    except Exception:
        return None


def _load_additional_trainables(model: torch.nn.Module, weight_path: str):
    non_svd_file = _resolve_weight_file(weight_path, 'non_lora_trainables.bin')
    if non_svd_file is None:
        rank0_print(f"No non-SVD trainables found at {weight_path}")
        return

    state = torch.load(non_svd_file, map_location='cpu')
    state = {(k[11:] if k.startswith('base_model.') else k): v for k, v in state.items()}
    if any(k.startswith('model.model.') for k in state):
        state = {(k[6:] if k.startswith('model.') else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        rank0_print(f"Warning: missing keys when loading non-SVD parameters: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        rank0_print(f"Warning: unexpected keys when loading non-SVD parameters: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.svd_enable:
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        # if training_args.svd_match_lora_rank is not None:
        #     inferred_groups = estimate_svd_num_groups_for_lora_equivalence(
        #         model, training_args.svd_match_lora_rank
        #     )
        #     rank0_print(
        #         "Auto-selecting svd_num_groups={} to match LoRA rank {} parameter counts".format(
        #             inferred_groups, training_args.svd_match_lora_rank
        #         )
        #     )
        #     training_args.svd_num_groups = inferred_groups
        rank0_print("Applying SVD-based adapters...")
        svd_config = SVDLinearConfig(
            num_groups=training_args.svd_num_groups,
            selected_group=training_args.svd_selected_group,
        )
        apply_svd_tuning(model, svd_config)
        inferred_dim = None
        for module in model.modules():
            if isinstance(module, LinearSVDAdapter):
                inferred_dim = module.adapter_dim
                break
        if inferred_dim is not None:
            rank0_print(
                f"SVD linear adapter dimension inferred from selected group: {inferred_dim}"
                f" (≈ min(in_features, out_features) / {training_args.svd_num_groups})"
            )
        model.config.svd_tuning = svd_config.to_dict()

        config_source = training_args.svd_weight_path
        if config_source:
            weight_locator = config_source if not os.path.isfile(config_source) else os.path.dirname(config_source)
            try:
                resolved_config = config_source
                loaded_config = load_svd_config(config_source)
            except (OSError, json.JSONDecodeError):
                resolved_config = _resolve_weight_file(weight_locator, 'svd_config.json')
                if resolved_config is None:
                    raise FileNotFoundError(f"Unable to locate svd_config.json under {config_source}")
                loaded_config = load_svd_config(resolved_config)
            if loaded_config.to_dict() != svd_config.to_dict():
                rank0_print("Warning: Loaded SVD config does not match training config. Using training config.")
            adapter_source = config_source
            if not (os.path.isdir(adapter_source) and os.path.exists(os.path.join(adapter_source, 'adapter_model.bin'))):
                resolved_adapter = _resolve_weight_file(weight_locator, 'adapter_model.bin')
                if resolved_adapter is not None:
                    adapter_source = resolved_adapter
            if not (os.path.isdir(adapter_source) or os.path.isfile(adapter_source)):
                raise FileNotFoundError(f"Unable to locate adapter_model.bin under {weight_locator}")
            load_svd_adapters(model, adapter_source)
            _load_additional_trainables(model, weight_locator)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.svd_enable:
        extra_trainable = []
        if model_args.tune_mm_mlp_adapter or training_args.mm_projector_lr is not None:
            extra_trainable.append('mm_projector')
        for name, param in model.named_parameters():
            if "svd_" in name or any(token in name for token in extra_trainable):
                param.requires_grad = True
            else:
                param.requires_grad = False

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.svd_enable:
        svd_state = get_svd_state_dict(model.named_parameters())
        non_svd_state = get_non_svd_state_dict(model.named_parameters())

        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            svd_config = SVDLinearConfig.from_dict(model.config.svd_tuning)
            os.makedirs(training_args.output_dir, exist_ok=True)
            torch.save(svd_state, os.path.join(training_args.output_dir, 'adapter_model.bin'))
            with open(os.path.join(training_args.output_dir, 'svd_config.json'), 'w') as f:
                json.dump(svd_config.to_dict(), f)
            torch.save(non_svd_state, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            generation_config = getattr(model, "generation_config", None)
            if generation_config is not None:
                try:
                    generation_config.save_pretrained(training_args.output_dir)
                except Exception as exc:
                    rank0_print(f"Warning: failed to save generation config: {exc}")
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()