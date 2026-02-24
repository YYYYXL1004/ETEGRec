import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from transformers import GenerationMixin
from torch import nn
from typing import Optional
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from vq import RQVAE
from layers import *


@dataclass
class QuantizeOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    rank_logits: Optional[torch.FloatTensor] = None
    seq_latents: Optional[torch.FloatTensor] = None
    seq_project_latents: Optional[torch.FloatTensor] = None
    dec_latents: Optional[torch.FloatTensor] = None


class Model(nn.Module, GenerationMixin):
    def __init__(self, config, model, n_items, code_length=1, code_number=256):
        super().__init__()
        self.model = model
        if hasattr(model, '_supports_cache_class'):
            self._supports_cache_class = model._supports_cache_class
        self.config = model.config
        self.base_model_prefix = "model"
        self.generation_config = model.generation_config
        self.main_input_name = model.main_input_name
        self.get_encoder = model.get_encoder
        self.device = model.device
        self.can_generate = lambda: True

        self.hidden_size = model.config.hidden_size
        self.n_items = n_items
        self.code_length = code_length
        self.code_number = code_number
        self.num_beams = config['num_beams']

        # 是否启用 cross-modal 双路模式
        self.cross_modal = config.get('cross_modal', False)

        if self.cross_modal:
            # 双路模式: text_semantic(collab+text=1024) + image_semantic(collab+image=1024)
            self.text_semantic_hidden_size = config.get('text_semantic_hidden_size')
            self.image_semantic_hidden_size = config.get('image_semantic_hidden_size')
            # 为了兼容 trainer.py 中 semantic_embedding 的引用，保留 semantic_hidden_size
            self.semantic_hidden_size = self.text_semantic_hidden_size

            self.text_semantic_embedding = nn.Embedding(self.n_items, self.text_semantic_hidden_size)
            self.text_semantic_embedding.requires_grad_(False)
            self.image_semantic_embedding = nn.Embedding(self.n_items, self.image_semantic_hidden_size)
            self.image_semantic_embedding.requires_grad_(False)

            # 双路 token_embeddings: 共享 T5 backbone，但 code→hidden 映射独立
            self.text_token_embeddings = nn.ModuleList([
                nn.Embedding(self.code_number, self.hidden_size) for _ in range(self.code_length)
            ])
            self.text_token_embeddings.requires_grad_(True)
            self.image_token_embeddings = nn.ModuleList([
                nn.Embedding(self.code_number, self.hidden_size) for _ in range(self.code_length)
            ])
            self.image_token_embeddings.requires_grad_(True)

            # 双路 adapter
            enc_adapter_dims = [self.hidden_size] + [config['e_dim']]
            self.text_enc_adapter = MLPLayers(layers=enc_adapter_dims)
            self.image_enc_adapter = MLPLayers(layers=enc_adapter_dims)

            dec_adapter_dims = [self.hidden_size] + [self.text_semantic_hidden_size]
            self.text_dec_adapter = MLPLayers(layers=dec_adapter_dims)
            dec_adapter_dims_img = [self.hidden_size] + [self.image_semantic_hidden_size]
            self.image_dec_adapter = MLPLayers(layers=dec_adapter_dims_img)
        else:
            # 原始单路模式
            self.semantic_hidden_size = config.get('semantic_hidden_size')
            self.semantic_embedding = nn.Embedding(self.n_items, self.semantic_hidden_size)
            self.semantic_embedding.requires_grad_(False)

            self.token_embeddings = nn.ModuleList([
                nn.Embedding(self.code_number, self.hidden_size) for _ in range(self.code_length)
            ])
            self.token_embeddings.requires_grad_(True)

            enc_adapter_layers = [self.hidden_size] + [config['e_dim']]
            self.enc_adapter = MLPLayers(layers=enc_adapter_layers)

            dec_adapter_layers = [self.hidden_size] + [self.semantic_hidden_size]
            self.dec_adapter = MLPLayers(layers=dec_adapter_layers)

        # parameters initialization
        self.apply(self._init_weights)

    def _get_token_embeddings(self, route=None):
        """根据 route 返回对应的 token_embeddings"""
        if self.cross_modal:
            if route == 'text':
                return self.text_token_embeddings
            elif route == 'image':
                return self.image_token_embeddings
            else:
                raise ValueError(f"cross_modal=True 时 route 必须为 'text' 或 'image', got {route}")
        else:
            return self.token_embeddings

    def _get_adapters(self, route=None):
        """根据 route 返回 (enc_adapter, dec_adapter)"""
        if self.cross_modal:
            if route == 'text':
                return self.text_enc_adapter, self.text_dec_adapter
            elif route == 'image':
                return self.image_enc_adapter, self.image_dec_adapter
            else:
                raise ValueError(f"cross_modal=True 时 route 必须为 'text' 或 'image', got {route}")
        else:
            return self.enc_adapter, self.dec_adapter

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, encoder_outputs=None, **kwargs):
        return {"decoder_input_ids": input_ids, "encoder_outputs": encoder_outputs, "attention_mask": attention_mask}

    def _shift_right(self, input_ids):
        pad_token_id = self.config.pad_token_id
        shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), pad_token_id, device=input_ids.device)
        shifted_input_ids = torch.cat([shifted_input_ids, input_ids], dim=-1)
        return shifted_input_ids

    def get_input_embeddings(self, input_ids, attention_mask, route=None):
        """
        将 code ids 映射为 hidden embeddings。
        route: 'text' / 'image' / None(单路模式)
        """
        token_embs = self._get_token_embeddings(route)
        attention_mask_flatten = attention_mask.reshape(-1)
        device = input_ids.device

        inputs_embeds = torch.zeros(*input_ids.shape, self.hidden_size, device=device)
        input_ids = input_ids.clone()
        input_ids[input_ids == -1] = 0
        for i in range(self.code_length):
            inputs_embeds[:, i::self.code_length] = token_embs[i](input_ids[:, i::self.code_length])

        inputs_embeds = inputs_embeds.view(-1, self.hidden_size)
        inputs_embeds[~attention_mask_flatten] = self.model.shared.weight[0].to(device)
        inputs_embeds = inputs_embeds.view(input_ids.shape[0], -1, self.hidden_size)

        return inputs_embeds

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, labels=None,
                decoder_input_ids=None, decoder_inputs_embeds=None, encoder_outputs=None,
                route=None, **kwargs):
        """
        route: 'text' / 'image' / None(单路模式)
        """
        token_embs = self._get_token_embeddings(route)
        enc_adapter, dec_adapter = self._get_adapters(route)

        if input_ids is not None:
            inputs_embeds = self.get_input_embeddings(input_ids, attention_mask, route=route)

        if decoder_input_ids is None and labels is None:
            decoder_input_ids = torch.zeros(input_ids.size(0), self.code_length).long().to(input_ids.device)
        elif decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)

        if decoder_inputs_embeds is None and decoder_input_ids is not None:
            decoder_inputs_embeds = []
            for i in range(min(decoder_input_ids.shape[1], self.code_length)):
                if i == 0:
                    code_embedding = self.model.shared
                else:
                    code_embedding = token_embs[i - 1]  # 0~255
                decoder_inputs_embeds.append(code_embedding(decoder_input_ids[:, i]))
            decoder_inputs_embeds = torch.stack(decoder_inputs_embeds, dim=1)

        model_outputs = self.model(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_hidden_states=True,
            encoder_outputs=encoder_outputs
        )

        decoder_outputs = model_outputs.decoder_hidden_states[-1]

        code_logits = []
        for i in range(min(decoder_inputs_embeds.shape[1], self.code_length)):
            centroid = token_embs[i].weight.t()
            code_logits.append(torch.matmul(decoder_outputs[:, i], centroid))
        code_logits = torch.stack(code_logits, dim=1)  # (batch, code_len, code_num)

        seq_latents = model_outputs.encoder_last_hidden_state.clone()
        seq_latents[~attention_mask] = 0
        seq_last_latents = torch.sum(seq_latents, dim=1) / attention_mask.sum(dim=1).unsqueeze(1)
        seq_project_latents = enc_adapter(seq_last_latents)

        dec_latents = model_outputs.decoder_hidden_states[-1].clone()
        dec_latents = dec_latents[:, 0, :]
        dec_latents = dec_adapter(dec_latents)

        outputs = QuantizeOutput(
            logits=code_logits,
            seq_latents=seq_last_latents,
            seq_project_latents=seq_project_latents,
            dec_latents=dec_latents
        )
        return outputs

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 n_return_sequences: int = 1, prefix_allowed_tokens_fn=None,
                 route=None) -> torch.Tensor:
        """
        Generates sequences using beam search algorithm.
        route: 'text' / 'image' / None(单路模式)
        """
        if prefix_allowed_tokens_fn is not None:
            inputs_embeds = self.get_input_embeddings(input_ids, attention_mask, route=route)
            outputs = super().generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_length=self.code_length + 1,
                num_beams=self.num_beams,
                num_return_sequences=n_return_sequences,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
            )
        else:
            outputs = self.my_beam_search(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.code_length + 1,
                num_beams=self.num_beams,
                num_return_sequences=n_return_sequences,
                return_score=False,
                route=route
            )
        outputs = outputs[:, 1:].reshape(-1, n_return_sequences, self.code_length)
        return outputs

    def my_beam_search(
        self,
        input_ids,
        attention_mask,
        max_length=6,
        num_beams=1,
        num_return_sequences=1,
        return_score=False,
        route=None
    ):
        """
        Adapted from huggingface's implementation.
        route: 'text' / 'image' / None(单路模式)
        """
        batch_size = input_ids.shape[0]

        input_ids, attention_mask, decoder_input_ids, beam_scores, beam_idx_offset = \
            self.prepare_beam_search_inputs(
                input_ids, attention_mask, batch_size, num_beams
            )

        inputs_embeds = self.get_input_embeddings(input_ids, attention_mask, route=route)

        with torch.no_grad():
            encoder_outputs = self.get_encoder()(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )

        while decoder_input_ids.shape[1] < max_length:
            with torch.no_grad():
                outputs = self.forward(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    route=route
                )

            decoder_input_ids, beam_scores = self.beam_search_step(
                outputs.logits,
                decoder_input_ids,
                beam_scores,
                beam_idx_offset,
                batch_size,
                num_beams
            )

        selection_mask = torch.zeros(batch_size, num_beams, dtype=bool)
        selection_mask[:, :num_return_sequences] = True

        if return_score:
            return decoder_input_ids[selection_mask.view(-1), :], \
                beam_scores[selection_mask.view(-1)] / (decoder_input_ids.shape[1] - 1)

        return decoder_input_ids[selection_mask.view(-1), :]

    def prepare_beam_search_inputs(self, input_ids, attention_mask, batch_size, num_beams):
        """Prepares and duplicates the input data for beam search decoding."""
        device = input_ids.device
        decoder_input_ids = torch.ones((batch_size * num_beams, 1), device=device, dtype=torch.long)
        initial_decoder_input_ids = decoder_input_ids * self.config.decoder_start_token_id

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9
        initial_beam_scores = beam_scores.view((batch_size * num_beams,))

        beam_idx_offset = torch.arange(batch_size, device=device).repeat_interleave(num_beams) * num_beams

        input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)

        return input_ids, attention_mask, initial_decoder_input_ids, initial_beam_scores, beam_idx_offset

    def beam_search_step(self, logits, decoder_input_ids, beam_scores, beam_idx_offset, batch_size, num_beams):
        """Executes one step of beam search."""
        assert batch_size * num_beams == logits.shape[0]

        vocab_size = logits.shape[-1]
        next_token_logits = logits[:, -1, :]
        next_token_scores = torch.log_softmax(next_token_logits, dim=-1)

        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
        next_token_scores, next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        beam_scores = next_token_scores[:, :num_beams].reshape(-1)
        beam_next_tokens = next_tokens[:, :num_beams].reshape(-1)
        beam_idx = next_indices[:, :num_beams].reshape(-1)

        decoder_input_ids = torch.cat([decoder_input_ids[beam_idx + beam_idx_offset, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        return decoder_input_ids, beam_scores

    def sample_generate(
        self,
        input_ids,
        attention_mask,
        num_samples=4,
        temperature=1.0,
        top_k=50,
        return_log_probs=False,
        route=None
    ):
        """
        采样生成多个候选序列，用于 GRPO 训练。
        route: 'text' / 'image' / None(单路模式)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        input_ids_expanded = input_ids.repeat_interleave(num_samples, dim=0)
        attention_mask_expanded = attention_mask.repeat_interleave(num_samples, dim=0)

        inputs_embeds = self.get_input_embeddings(input_ids_expanded, attention_mask_expanded, route=route)

        with torch.no_grad():
            encoder_outputs = self.get_encoder()(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask_expanded,
                return_dict=True
            )

        decoder_input_ids = torch.full(
            (batch_size * num_samples, 1),
            self.config.decoder_start_token_id,
            dtype=torch.long,
            device=device
        )

        all_log_probs = [] if return_log_probs else None

        for step in range(self.code_length):
            with torch.no_grad():
                outputs = self.forward(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask_expanded,
                    decoder_input_ids=decoder_input_ids,
                    route=route
                )

            next_token_logits = outputs.logits[:, -1, :] / temperature

            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            if return_log_probs:
                log_probs_step = torch.log_softmax(next_token_logits, dim=-1)
                selected_log_probs = log_probs_step.gather(dim=-1, index=next_tokens)
                all_log_probs.append(selected_log_probs)

            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)

        gen_codes = decoder_input_ids[:, 1:]
        gen_codes = gen_codes.view(batch_size, num_samples, self.code_length)

        if return_log_probs:
            seq_log_probs = torch.cat(all_log_probs, dim=-1).sum(dim=-1)
            seq_log_probs = seq_log_probs.view(batch_size, num_samples)
            return gen_codes, seq_log_probs

        return gen_codes

    def compute_log_probs(self, input_ids, attention_mask, target_codes, route=None):
        """
        计算给定 target_codes 的 log probability (Teacher Forcing)
        route: 'text' / 'image' / None(单路模式)
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_codes,
            route=route
        )

        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=target_codes.unsqueeze(-1)).squeeze(-1)
        seq_log_probs = token_log_probs.sum(dim=-1)

        return seq_log_probs
