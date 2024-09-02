CITATION = """
Implementation reference: https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/models/rag/modeling_rag.py
Retrieval-Augmented Generation paper link: https://arxiv.org/abs/2005.11401
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import ModelOutput


class RagSequenceModel(T5ForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        doc_scores: Optional[torch.LongTensor] = None,
        n_docs: int = None,
        add_eos_loss: bool = None,
        reduction: str = None,
    ) -> ModelOutput:

        assert (
            doc_scores is not None
        ), "Make sure that `doc_scores` are passed when passing `encoder_outputs` to the forward function."

        assert (doc_scores.shape[1] % n_docs) == 0, (
            f" The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is"
            f" {input_ids.shape[0]}."
        )

        # Decoder input without context documents
        target = decoder_input_ids.clone().detach().to(decoder_input_ids)
        decoder_input_ids = decoder_input_ids.repeat_interleave(n_docs, dim=0)

        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.repeat_interleave(n_docs, dim=0)

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        loss = None
        if labels is not None:
            loss = self.get_nll(
                seq_logits=outputs.logits,
                doc_scores=doc_scores,
                target=target,
                reduce_loss=False,
                epsilon=0.0,
                n_docs=n_docs,
                add_eos_loss=add_eos_loss,
                reduction=reduction,
            )
        return ModelOutput(
            loss=loss,
            logits=outputs.logits,
        )

    def get_nll(
        self,
        seq_logits,
        doc_scores,
        target,
        reduce_loss=False,
        epsilon=0.0,
        n_docs=None,
        add_eos_loss=False,
        reduction="sum",
    ):
        # shift tokens left
        filled_id = self.config.eos_token_id if add_eos_loss else self.config.pad_token_id
        target = torch.cat([target[:, 1:], target.new(target.shape[0], 1).fill_(filled_id)], 1)

        def _mask_pads(ll, smooth_obj):
            pad_mask = target.eq(self.config.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        # seq_logits dim = (batch*n_docs, tgt_len , #vocabs)
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
        )  # batch_size x n_docs x tgt_len x #vocab_size
        doc_logprobs = nn.functional.log_softmax(doc_scores, dim=1).unsqueeze(-1).unsqueeze(-1)

        # RAG-sequence marginalization
        first_token_scores = seq_logprobs[:, :, :1, :]
        second_token_scores = seq_logprobs[:, :, 1:2, :]
        remainder = seq_logprobs[:, :, 2:, :]
        rag_logprobs = torch.cat([first_token_scores, second_token_scores + doc_logprobs, remainder], dim=2)

        # calculate loss
        target = target.unsqueeze(1).unsqueeze(-1).repeat(1, n_docs, 1, 1)
        assert target.dim() == rag_logprobs.dim()

        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits

        ll, smooth_obj = _mask_pads(ll, smooth_obj)

        # sum over tokens
        if reduction == "sum":
            ll = ll.sum(2)
            smooth_obj = smooth_obj.sum(2)
            ll = ll.logsumexp(1)  # logsumexp over docs
            smooth_obj = smooth_obj.logsumexp(1)
        elif reduction == "mean" and add_eos_loss:
            ll = ll.mean(2)
            smooth_obj = smooth_obj.mean(2)
            ll = ll.logsumexp(1)  # logsumexp over docs
            smooth_obj = smooth_obj.logsumexp(1)
        elif reduction == "mean" and not add_eos_loss:
            ll = ll[:, :, :-1].mean(2)
            smooth_obj = smooth_obj[:, :, :-1].mean(2)
            ll = ll.logsumexp(1)  # logsumexp over docs
            smooth_obj = smooth_obj.logsumexp(1)
        else:
            raise NotImplementedError()

        nll_loss = -ll
        smooth_loss = -smooth_obj

        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss


class RagTokenModel(T5ForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def marginalize(self, seq_logits, doc_scores, n_docs=None):

        # RAG-token marginalization
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
        )
        doc_logprobs = torch.log_softmax(doc_scores, dim=1)
        log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1).unsqueeze(-1)
        return torch.logsumexp(log_prob_sum, dim=1)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        doc_scores: Optional[torch.LongTensor] = None,
        n_docs: Optional[int] = None,
        add_eos_loss: Optional[bool] = False,
        reduction: Optional[str] = "sum",
        **kwargs,  # needs kwargs for generation
    ) -> ModelOutput:
        assert (
            doc_scores is not None
        ), "Make sure that `doc_scores` are passed when passing `encoder_outputs` to the forward function."

        assert (doc_scores.shape[1] % n_docs) == 0, (
            f" The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is"
            f" {input_ids.shape[0]}."
        )

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels

        target = decoder_input_ids.clone().detach().to(decoder_input_ids)
        decoder_input_ids = decoder_input_ids.repeat_interleave(n_docs, dim=0)

        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.repeat_interleave(n_docs, dim=0)

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        loss = None
        logits = outputs.logits
        if labels is not None:
            loss = self.get_nll(
                seq_logits=outputs.logits,
                doc_scores=doc_scores,
                target=target,
                reduce_loss=False,
                epsilon=0.0,
                n_docs=n_docs,
                add_eos_loss=add_eos_loss,
                reduction=reduction,
            )

        return ModelOutput(
            loss=loss,
            logits=logits,
        )

    def get_nll(
        self,
        seq_logits,
        doc_scores,
        target,
        reduce_loss=False,
        epsilon=0.0,
        n_docs=None,
        add_eos_loss=False,
        reduction="sum",
    ):

        # shift tokens left
        filled_id = self.config.eos_token_id if add_eos_loss else self.config.pad_token_id
        target = torch.cat([target[:, 1:], target.new(target.shape[0], 1).fill_(filled_id)], 1)

        def _mask_pads(ll, smooth_obj):
            pad_mask = target.eq(self.config.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        rag_logprobs = self.marginalize(seq_logits, doc_scores, n_docs)

        target = target.unsqueeze(-1)
        assert target.dim() == rag_logprobs.dim()

        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits
        ll, smooth_obj = _mask_pads(ll, smooth_obj)
        if reduction == "sum":
            ll = ll.sum(1)  # sum over tokens
            smooth_obj = smooth_obj.sum(1)
        elif reduction == "mean" and add_eos_loss:
            ll = ll.mean(1)
            smooth_obj = smooth_obj.mean(1)
        elif reduction == "mean" and not add_eos_loss:
            ll = ll[:, :-1].mean(1)
            smooth_obj = smooth_obj[:, :-1].mean(1)
        else:
            raise NotImplementedError()

        nll_loss = -ll
        smooth_loss = -smooth_obj

        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss


    def generate(self, input_ids, attention_mask, decoder_input_ids, **generation_kwargs):
        max_new_tokens = generation_kwargs.pop("max_new_tokens")
        do_sample = generation_kwargs.pop("do_sample")
        if do_sample:
            raise NotImplementedError("Sampling is not implemented for RagTokenModel")
        
        valid_max_sequence_length = torch.ones_like(decoder_input_ids).sum(-1).max().item()
        decoder_input_ids = decoder_input_ids[:, -valid_max_sequence_length :]

        buffer_ids = decoder_input_ids

        num_generated_tokens = 0
        stop_token_id = self.config.eos_token_id
        output_ids = decoder_input_ids.long().to(decoder_input_ids.device)
        buffer_next_token_id = None
        
        # greedy decoding
        # TODO: add past_key_value implementation for caching
        while num_generated_tokens < max_new_tokens:
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    doc_scores=torch.zeros(1, input_ids.size(0)).float(),
                    n_docs=input_ids.size(0),
                )
                logits = outputs.logits.mean(dim=0)
                next_token_ids = logits[-1, :].argmax(-1).unsqueeze(0).unsqueeze(0)
                buffer_next_token_id = next_token_ids[:, -1].squeeze().item()
                # Stop at eos token
                if buffer_next_token_id == stop_token_id:
                    break
                output_ids = torch.cat((output_ids, next_token_ids), dim=-1)
                buffer_ids = torch.cat((buffer_ids, next_token_ids), dim=-1)
                # Update decoder_input_ids
                decoder_input_ids = torch.cat((decoder_input_ids, next_token_ids), dim=-1)
                valid_max_sequence_length += 1
                num_generated_tokens += 1

        return output_ids
