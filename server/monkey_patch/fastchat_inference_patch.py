"""Inference for FastChat models."""

import gc

from typing import Iterable, Dict
import warnings


import fastchat.serve
import fastchat.serve
import fastchat.serve.inference
import torch


from fastchat.model.model_adapter import (
    load_model,
    get_conversation_template,
    get_generate_stream_function,
)
from fastchat.modules.awq import AWQConfig
from fastchat.modules.gptq import GptqConfig
from fastchat.modules.exllama import ExllamaConfig
from fastchat.modules.xfastertransformer import XftConfig
from fastchat.utils import is_partial_stop, is_sentence_complete, get_context_length
from fastchat.serve.inference import prepare_logits_processor




def inferenc_patch():
    pass

    @torch.inference_mode()
    def generate_stream(
        model,
        tokenizer,
        params: Dict,
        device: str,
        context_len: int,
        stream_interval: int = 2,
        judge_sent_end: bool = False,
    ):
        if hasattr(model, "device"):
            device = model.device

        # Read parameters
        prompt = params["prompt"]
        len_prompt = len(prompt)
        temperature = float(params.get("temperature", 1.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = int(params.get("top_k", -1))  # -1 means disable
        max_new_tokens = int(params.get("max_new_tokens", 256))
        logprobs = params.get("logprobs", None)  # FIXME: Support logprobs>1.
        echo = bool(params.get("echo", True))
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_token_ids", None) or []
        if tokenizer.eos_token_id not in stop_token_ids:
            stop_token_ids.append(tokenizer.eos_token_id)

        logits_processor = prepare_logits_processor(
            temperature, repetition_penalty, top_p, top_k
        )
        input_ids = tokenizer(prompt).input_ids

        if model.config.is_encoder_decoder:
            max_src_len = context_len
        else:  # truncate
            # max_src_len = context_len - max_new_tokens - 1
            max_src_len = context_len - max_new_tokens

        input_ids = input_ids[-max_src_len:]
        output_ids = list(input_ids)
        input_echo_len = len(input_ids)

        if model.config.is_encoder_decoder:
            if logprobs is not None:  # FIXME: Support logprobs for encoder-decoder models.
                raise NotImplementedError
            encoder_output = model.encoder(
                input_ids=torch.as_tensor([input_ids], device=device)
            )[0]
            start_ids = torch.as_tensor(
                [[model.generation_config.decoder_start_token_id]],
                dtype=torch.int64,
                device=device,
            )
        else:
            start_ids = torch.as_tensor([input_ids], device=device)

        past_key_values = out = None
        token_logprobs = [None]  # The first token has no logprobs.
        sent_interrupt = False
        finish_reason = None
        stopped = False
        for i in range(max_new_tokens):
            if i == 0:  # prefill
                if model.config.is_encoder_decoder:
                    out = model.decoder(
                        input_ids=start_ids,
                        encoder_hidden_states=encoder_output,
                        use_cache=True,
                    )
                    logits = model.lm_head(out[0])
                else:
                    out = model(input_ids=start_ids, use_cache=True)
                    logits = out.logits
                past_key_values = out.past_key_values

                if logprobs is not None:
                    # Prefull logprobs for the prompt.
                    shift_input_ids = start_ids[..., 1:].contiguous()
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_logits = torch.log_softmax(shift_logits, dim=-1).tolist()
                    for label_id, logit in zip(
                        shift_input_ids[0].tolist(), shift_logits[0]
                    ):
                        token_logprobs.append(logit[label_id])
            else:  # decoding
                if model.config.is_encoder_decoder:
                    out = model.decoder(
                        input_ids=torch.as_tensor(
                            [[token] if not sent_interrupt else output_ids],
                            device=device,
                        ),
                        encoder_hidden_states=encoder_output,
                        use_cache=True,
                        past_key_values=past_key_values if not sent_interrupt else None,
                    )
                    sent_interrupt = False

                    logits = model.lm_head(out[0])
                else:
                    out = model(
                        input_ids=torch.as_tensor(
                            [[token] if not sent_interrupt else output_ids],
                            device=device,
                        ),
                        use_cache=True,
                        past_key_values=past_key_values if not sent_interrupt else None,
                    )
                    sent_interrupt = False
                    logits = out.logits
                past_key_values = out.past_key_values

            if logits_processor:
                if repetition_penalty > 1.0:
                    tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
                else:
                    tmp_output_ids = None
                last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
            else:
                last_token_logits = logits[0, -1, :]

            if device == "mps":
                # Switch to CPU by avoiding some bugs in mps backend.
                last_token_logits = last_token_logits.float().to("cpu")

            if temperature < 1e-5 or top_p < 1e-8:  # greedy
                _, indices = torch.topk(last_token_logits, 2)
                tokens = [int(index) for index in indices.tolist()]
            else:
                probs = torch.softmax(last_token_logits, dim=-1)
                indices = torch.multinomial(probs, num_samples=2)
                tokens = [int(token) for token in indices.tolist()]
            token = tokens[0]
            output_ids.append(token)
            if logprobs is not None:
                # Cannot use last_token_logits because logprobs is based on raw logits.
                token_logprobs.append(
                    torch.log_softmax(logits[0, -1, :], dim=-1)[token].tolist()
                )

            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            # Yield the output tokens
            if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                if echo:
                    tmp_output_ids = output_ids
                    rfind_start = len_prompt
                else:
                    tmp_output_ids = output_ids[input_echo_len:]
                    rfind_start = 0

                output = tokenizer.decode(
                    tmp_output_ids,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
                ret_logprobs = None
                if logprobs is not None:
                    ret_logprobs = {
                        "text_offset": [],
                        "tokens": [
                            tokenizer.decode(token)
                            for token in (
                                output_ids if echo else output_ids[input_echo_len:]
                            )
                        ],
                        "token_logprobs": token_logprobs
                        if echo
                        else token_logprobs[input_echo_len:],
                        "top_logprobs": [{}]
                        * len(token_logprobs if echo else token_logprobs[input_echo_len:]),
                    }
                    # Compute text_offset
                    curr_pos = 0
                    for text in ret_logprobs["tokens"]:
                        ret_logprobs["text_offset"].append(curr_pos)
                        curr_pos += len(text)

                # TODO: For the issue of incomplete sentences interrupting output, apply a patch and others can also modify it to a more elegant way
                if judge_sent_end and stopped and not is_sentence_complete(output):
                    if len(tokens) > 1:
                        token = tokens[1]
                        output_ids[-1] = token
                    else:
                        output_ids.pop()
                    stopped = False
                    sent_interrupt = True

                partially_stopped = False
                if stop_str:
                    if isinstance(stop_str, str):
                        pos = output.rfind(stop_str, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                        else:
                            partially_stopped = is_partial_stop(output, stop_str)
                    elif isinstance(stop_str, Iterable):
                        for each_stop in stop_str:
                            pos = output.rfind(each_stop, rfind_start)
                            if pos != -1:
                                output = output[:pos]
                                stopped = True
                                break
                            else:
                                partially_stopped = is_partial_stop(output, each_stop)
                                if partially_stopped:
                                    break
                    else:
                        raise ValueError("Invalid stop field type.")

                # Prevent yielding partial stop sequence
                if not partially_stopped:
                    yield {
                        "text": output,
                        "logprobs": ret_logprobs,
                        "usage": {
                            "prompt_tokens": input_echo_len,
                            "completion_tokens": i,
                            "total_tokens": input_echo_len + i,
                        },
                        "finish_reason": None,
                    }

            if stopped:
                break

        # Finish stream event, which contains finish reason
        else:
            finish_reason = "length"

        if stopped:
            finish_reason = "stop"

        yield {
            "text": output,
            "logprobs": ret_logprobs,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": i,
                "total_tokens": input_echo_len + i,
            },
            "finish_reason": finish_reason,
        }

        # Clean
        del past_key_values, out
        gc.collect()
        torch.cuda.empty_cache()
        if device == "xpu":
            torch.xpu.empty_cache()
        if device == "npu":
            torch.npu.empty_cache()

    import fastchat
    fastchat.serve.inference.generate_stream = generate_stream

