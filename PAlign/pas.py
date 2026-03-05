"""
Model Setup and Inference Script for Meta-LLaMA

This script defines a class for loading and interacting with the Meta-LLaMA model,
including handling model activations and generating responses. The script includes
methods for setting up the model, processing input datasets, and performing inference
with various configurations.
"""

import json
import random
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
from einops import rearrange
import pickle
from functools import partial
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.rich import tqdm
import matplotlib
from copy import deepcopy


def _chat_ids(tokenizer, messages, **kwargs):
    """v4/v5 compat: apply_chat_template returning plain input_ids list."""
    out = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(out, list):
        return out
    return out.input_ids


class _ModuleInputCapture:
    """Capture inputs to named submodules via forward pre-hooks."""

    def __init__(self, model, layer_names):
        self._handles, self._outputs = [], {}
        modules = dict(model.named_modules())
        for name in layer_names:
            self._handles.append(
                modules[name].register_forward_pre_hook(self._make_hook(name))
            )

    def _make_hook(self, name):
        def hook(module, args):
            self._outputs[name] = args[0]
        return hook

    def __enter__(self):
        return self

    def __getitem__(self, key):
        return self._outputs[key]

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()


def get_model(model_name='meta-llama/Llama-2-7b-chat-hf'):
    """
    Loads and sets up the Meta-LLaMA model for inference and activation handling.

    Args:
    model_name (str): Name of the model to load.

    Returns:
    model: Configured LLaMA model.
    tokenizer: Corresponding tokenizer.
    """


    class PASLM:
        """
        Main class for loading, configuring, and interacting with the Meta-LLaMA model.
        """
        def __init__(self):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model_file = model_name

            num_gpus = torch.cuda.device_count()
            dmap = "auto" if num_gpus > 1 else "cuda"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="auto", device_map=dmap, low_cpu_mem_usage=True)
            self.model.eval()
            self.device = self.model.device

            self.bias_cache = []
            for i, layer in enumerate(self.model.model.layers):
                self.bias_cache.append(deepcopy(self.model.model.layers[i].self_attn.o_proj.bias))

        def generate(model, text, max_length=512, max_new_tokens=None):
            """
            Generates responses based on the input text using the model.

            Args:
            model: The model to use for generation.
            text (str): Input text for the model.
            max_length (int): Maximum length of the generated text.
            max_new_tokens (int): Maximum number of new tokens to generate.

            Returns:
            tokens: Generated tokens.
            """
            tokenizer = model.tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            stop_id = tokenizer.sep_token_id
            pad_id = tokenizer.pad_token_id

            device = model.device
            gc = model.model.generation_config or GenerationConfig.from_pretrained(model.model_file)
            eos_ids = gc.eos_token_id or model.tokenizer.eos_token_id
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]
            eos_ids_tensor = torch.tensor(eos_ids, device=device)
            input_ids = [t for t in text]
            min_prompt_len = min(len(t) for t in input_ids)
            max_prompt_len = max(len(t) for t in input_ids)

            if max_new_tokens:
                max_length = max_prompt_len + max_new_tokens
            tokens = torch.full((len(input_ids), max_length), pad_id, dtype=torch.long).to(device)
            for k, t in enumerate(input_ids):
                tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
            prev_pos = 0
            cur_pos = min_prompt_len - 1
            input_text_mask = tokens != pad_id
            eos_reached = torch.tensor([False] * len(input_ids), device=device)
            past_key_values = None

            with torch.no_grad():
                for cur_pos_add in range(max_length):
                    cur_pos += 1
                    if prev_pos != 0:
                        prev_pos = cur_pos - 1
                    if tokens.shape[1] == cur_pos:
                        break
                    torch.cuda.empty_cache()

                    logits = model.model(tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=past_key_values)
                    next_token = torch.topk(logits['logits'][:, -1], 1, dim=-1)[1][:, -1]
                    next_token = next_token.reshape(-1)
                    next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
                    # Replace generated tokens with pad for sequences that already hit EOS
                    next_token = torch.where(eos_reached & ~input_text_mask[:, cur_pos],
                                             torch.tensor(pad_id, device=device), next_token)
                    tokens[:, cur_pos] = next_token
                    eos_reached |= (~input_text_mask[:, cur_pos]) & torch.isin(next_token, eos_ids_tensor)

                    if all(eos_reached):
                        break
                    prev_pos = cur_pos
                    past_key_values = logits["past_key_values"]
            return tokens

        def __call__(self, input_ids):
            """
            Calls the model with the given input IDs.

            Args:
            input_ids: Input IDs for the model.

            Returns:
            logits: Model output logits.
            """
            with torch.no_grad():
                logits = self.model(input_ids)
                return logits

        def get_last_activations(self, layer):
            """
            Retrieves the last activations for a specific layer.

            Args:
            layer: Layer number.

            Returns:
            activations: Activations of the specified layer.
            """
            return self.model.model.layers[layer].activations

        def reset_all(self):
            """
            Resets all internal states and biases of the model.
            """
            for i, layer in enumerate(self.model.model.layers):
                self.model.model.layers[i].self_attn.o_proj.bias = self.bias_cache[i]

        def get_activations(self, all_head_wise_activations, labels, num_to_intervene=48, val_ratio=0.4):
            """
            Gets the activations for the model based on the given head-wise activations and labels.

            Args:
            all_head_wise_activations: All head-wise activations for both pos and neg pairs of current sample.
            labels: Labels for the activations for if it fits current item's pos or neg pair.
            num_to_intervene: Number of heads to intervene.
            """
            def get_top_heads(separated_activations, separated_labels, num_layers, num_heads, num_to_intervene):

                probes, all_head_accs_np = train_probes(separated_activations,
                                                        separated_labels, num_layers=num_layers, num_heads=num_heads)
                all_head_accs_np = all_head_accs_np.reshape(num_layers, num_heads,2)
                all_head_accs_np = all_head_accs_np.mean(2) # so it is take the mean accuracy of training and validation sets...
                top_accs = np.argsort(all_head_accs_np.reshape(num_heads * num_layers))[::-1][:num_to_intervene]
                top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]

                return top_heads, probes

            def train_probes(separated_head_wise_activations, separated_labels,
                             num_layers, num_heads):

                all_head_accs = []
                probes = []

                train_idxs = np.arange(len(separated_labels))

                # pick a val set using numpy
                rng = np.random.RandomState(42)
                train_set_idxs = rng.choice(train_idxs, size=int(len(train_idxs) * (1 - val_ratio)),
                                            replace=False)
                val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

                all_X_train = np.array([separated_head_wise_activations[i] for i in train_set_idxs])
                all_X_val = np.array([separated_head_wise_activations[i] for i in val_set_idxs])
                y_train = np.array([separated_labels[i] for i in train_set_idxs])
                y_val = np.array([separated_labels[i] for i in val_set_idxs])

                for layer in tqdm(range(num_layers)):
                    for head in range(num_heads):
                        X_train = all_X_train[:, layer, head, :]
                        X_val = all_X_val[:, layer, head, :]

                        clf = LogisticRegression(random_state=42, max_iter=1000).fit(X_train, y_train)
                        y_pred = clf.predict(X_train)
                        y_val_pred = clf.predict(X_val)
                        all_head_accs.append([accuracy_score(y_val, y_val_pred),accuracy_score(y_train,y_pred)])
                        probes.append(clf)

                all_head_accs_np = np.array(all_head_accs)

                return probes, all_head_accs_np

            def flattened_idx_to_layer_head(flattened_idx, num_heads):
                return flattened_idx // num_heads, flattened_idx % num_heads

            def layer_head_to_flattened_idx(layer, head, num_heads):
                return layer * num_heads + head

            def get_interventions_dict(top_heads, probes, tuning_activations, num_heads,com_directions=None):

                interventions = {}
                for layer, head in top_heads:
                    interventions[f"model.layers.{layer}.self_attn.o_proj"] = []
                for layer, head in top_heads:
                    if com_directions is not None:
                        direction = com_directions[layer_head_to_flattened_idx(layer, head, num_heads)]
                    else:
                        direction = probes[layer_head_to_flattened_idx(layer, head, num_heads)].coef_
                    direction = direction / np.linalg.norm(direction)
                    activations = tuning_activations[:, layer, head, :]  # batch x 128
                    proj_vals = activations @ direction.T
                    proj_val_std = np.std(proj_vals)
                    interventions[f"model.layers.{layer}.self_attn.o_proj"].append(
                        (head, direction.squeeze(), proj_val_std))
                for layer, head in top_heads:
                    interventions[f"model.layers.{layer}.self_attn.o_proj"] = sorted(
                        interventions[f"model.layers.{layer}.self_attn.o_proj"], key=lambda x: x[0])

                return interventions

            def get_com_directions(num_layers, num_heads,usable_head_wise_activations,
                                   usable_labels):

                com_directions = []

                for layer in range(num_layers):
                    for head in range(num_heads):
                       # usable_idxs = np.concatenate([train_set_idxs, val_set_idxs], axis=0)
                        #usable_head_wise_activations = np.concatenate(
                            #[separated_head_wise_activations[i][:, layer, head, :] for i in usable_idxs], axis=0)
                        #usable_labels = np.concatenate([separated_labels[i] for i in usable_idxs], axis=0)
                        usable_labels = np.array(usable_labels)
                        head_wise_activations = usable_head_wise_activations[:,layer, head, :]
                        # so here's the average activation of pos activations for current layer's head; average across all pos pairs
                        true_mass_mean = np.mean(head_wise_activations[usable_labels == 1], axis=0)
                        # this is neg activation; average across all neg pairs
                        false_mass_mean = np.mean(head_wise_activations[usable_labels == 0], axis=0)
                        # and finally append the differences together, so that we can store the case for each layer's head
                        com_directions.append(true_mass_mean - false_mass_mean) 
                com_directions = np.array(com_directions)

                return com_directions

            num_layers = self.model.model.config.num_hidden_layers
            num_heads = self.model.model.config.num_attention_heads

            head_wise_activations = deepcopy(all_head_wise_activations)
            head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h=num_heads)
            tuning_activations = deepcopy(all_head_wise_activations)
            tuning_activations = rearrange(tuning_activations, 'b l (h d) -> b l h d', h=num_heads)

            top_heads, probes = get_top_heads(head_wise_activations, labels, num_layers, num_heads, num_to_intervene)

            com_directions = get_com_directions(num_layers, num_heads, head_wise_activations,
                                                labels)

            interventions = get_interventions_dict(top_heads, probes, tuning_activations, num_heads,com_directions)

            return interventions

        def preprocess_activate_dataset(self, dataset, system_prompt="You are a helpful, honest and concise assistant."):
            """
            Preprocesses the dataset and retrieves activations for the model.

            Args:
            dataset: Input dataset.
            system_prompt (str): System prompt for the model.

            Returns:
            all_head_wise_activations: All head-wise activations.
            """
            self.system_prompt = system_prompt

            def prompt_to_tokens(tokenizer, system_prompt, instruction, model_output):
                if model_output:
                    con = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": model_output}
                    ]
                    return torch.tensor(_chat_ids(tokenizer, con)[:-1]).unsqueeze(0)
                else:
                    con = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": instruction},
                    ]
                    return torch.tensor(_chat_ids(tokenizer, con)).unsqueeze(0)

            def data_preprocess(dataset):
                all_prompts = []
                for i in range(len(dataset)):
                    question = dataset[i]['question']

                    pos_answer = dataset[i]['answer_matching_behavior']
                    pos_tokens = prompt_to_tokens(self.tokenizer, self.system_prompt, question, pos_answer)
                    all_prompts.append(pos_tokens)

                    neg_answer = dataset[i]['answer_not_matching_behavior']
                    neg_tokens = prompt_to_tokens(self.tokenizer, self.system_prompt, question, neg_answer)
                    all_prompts.append(neg_tokens)

                return all_prompts

            def get_llama_activations_bau(model, prompt):
                HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]

                with torch.no_grad():
                    prompt = prompt.to(model.device)
                    with _ModuleInputCapture(model, HEADS) as ret:
                        output = model(prompt, output_hidden_states=True)
                    hidden_states = output.hidden_states
                    hidden_states = torch.stack(hidden_states, dim=0).squeeze().to(torch.float16).detach().cpu().numpy()
                    head_wise_hidden_states = [ret[head].squeeze().to(torch.float16).detach().cpu() for head in HEADS]
                    head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()

                return hidden_states, head_wise_hidden_states

            prompts = data_preprocess(dataset)

            all_layer_wise_activations = []
            all_head_wise_activations = []

            for prompt in tqdm(prompts):
                layer_wise_activation, head_wise_activation = get_llama_activations_bau(self.model, prompt)
                all_layer_wise_activations.append(layer_wise_activation[:, -1, :])
                all_head_wise_activations.append(head_wise_activation[:, -1, :])

            return all_head_wise_activations

        def set_activate(self, interventions, alpha):
            """
            Sets the activations for the model based on interventions.

            Args:
            interventions: Dictionary of interventions.
            alpha: Activation strength.
            """
            num_layers = self.model.model.config.num_hidden_layers
            num_heads = self.model.model.config.num_attention_heads

            head_dim = getattr(self.model.model.config, 'head_dim',
                               self.model.model.config.hidden_size // num_heads)
            for head_out_name, list_int_vec in interventions.items():
                layer_no = int(head_out_name.split('.')[2])
                displacement = np.zeros((num_heads, head_dim))
                for head_no, head_vec, std in list_int_vec:
                    displacement[head_no] = alpha * std * head_vec
                weight = self.model.model.layers[layer_no].self_attn.o_proj.weight
                device = weight.device
                displacement = torch.tensor(rearrange(displacement, 'h d -> (h d)'), device=device)
                bias_tobe = F.linear(displacement.to(weight.dtype), weight)
                original_bias = self.bias_cache[layer_no]
                if original_bias is not None:
                    bias_tobe = bias_tobe + original_bias.to(device=device, dtype=bias_tobe.dtype)
                self.model.model.layers[layer_no].self_attn.o_proj.bias = torch.nn.parameter.Parameter(bias_tobe)

    model = PASLM()
    model.reset_all()
    return model, model.tokenizer
