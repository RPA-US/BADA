import gc
from multiprocessing import Process, Queue
from typing import Any, Dict, List, Tuple

import torch
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.language_models.llms import LLM
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

#######
# INTERFACES
#######


class ModelInterface:
    capabilities: List[str]
    skip_special_tokens: bool = False

    def __init__(self) -> None:
        pass

    def inference(
        self, sys_prompt: str, user_prompt: str, *args: Any, **kwargs: Any
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Performs an inference using qwen-vl based models

        @returns prompt messages and response
        """
        pass

    def load_model(self) -> Tuple[Any, ...]:
        """
        Loads and returns the model to make inferences on
        """
        pass

    def unload(self, *args: Any) -> None:
        """
        Unloads the given items and extras from cuda memory
        """
        try:
            for arg in args:
                del arg
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


#######
# IMPLEMENTATION
#######


# TODO: Convert to LangChain
class QwenVLLangChain(ModelInterface, LLM):
    capabilities: List[str] = ["image", "text"]
    chat_template: str = (
        "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )

    def __init__(
        self,
        model_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.model_name: str = model_name

    def multiprocess_inference(
        self,
        sys_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        model, processor = self.load_model()

        messages: List[Dict[str, Any]] = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        messages.append(
            {
                "role": "user",
                "content": [
                    *[
                        {
                            "type": t,
                            t: val,
                        }
                        for t, val in kwargs.items()
                        if t in self.capabilities
                    ],
                    {"type": "text", "text": user_prompt},
                ],
            },
        )

        text: str = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)

        generated_ids_trimmed: List[torch.Tensor] = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        processed_output_text: str = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=self.skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

        model = processor = inputs = generated_ids = generated_ids_trimmed = None
        self.unload(model, processor, inputs, generated_ids, generated_ids_trimmed)

        kwargs["result_queue"].put((messages, processed_output_text))

    def inference(
        self,
        sys_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[list[Dict[str, Any]], str]:
        """
        Performs an inference using qwen-vl based models
        """
        # This is strictly necessary to ensure ALL memory held by torch is released when the inference is done
        # Running the inference without this results in many dangling tensors for some reason
        result_queue = Queue()
        kwargs["result_queue"] = result_queue
        p = Process(
            target=self.multiprocess_inference,
            args=(sys_prompt, user_prompt, max_tokens, *args),
            kwargs=kwargs,
        )
        p.start()
        p.join()

        # result_queue.get() function as a pop. Always save it to a variable or return it directly
        return result_queue.get()

    def load_model(self) -> Tuple[Any]:
        """
        Loads and returns the model to make inferences on
        """
        model = LlamaCpp(
            model_path="./models/Qwen2-VL-7B-Instruct-Q5_K_M.gguf",
            n_gpu_layers=1,
            n_batch=512,
            n_ctx=2048,
            f16_kv=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=True,
        )
        return model


class QwenVLModel(ModelInterface):
    capabilities: List[str] = ["image", "text"]

    def __init__(
        self,
        model_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.model_name: str = model_name

    def multiprocess_inference(
        self,
        sys_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        model, processor = self.load_model()

        messages: List[Dict[str, Any]] = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        messages.append(
            {
                "role": "user",
                "content": [
                    *[
                        {
                            "type": t,
                            t: val,
                        }
                        for t, val in kwargs.items()
                        if t in self.capabilities
                    ],
                    {"type": "text", "text": user_prompt},
                ],
            },
        )

        text: str = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)

        generated_ids_trimmed: List[torch.Tensor] = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        processed_output_text: str = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=self.skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

        model = processor = inputs = generated_ids = generated_ids_trimmed = None
        self.unload(model, processor, inputs, generated_ids, generated_ids_trimmed)

        kwargs["result_queue"].put((messages, processed_output_text))

    def inference(
        self,
        sys_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[list[Dict[str, Any]], str]:
        """
        Performs an inference using qwen-vl based models
        """
        # This is strictly necessary to ensure ALL memory held by torch is released when the inference is done
        # Running the inference without this results in many dangling tensors for some reason
        result_queue = Queue()
        kwargs["result_queue"] = result_queue
        p = Process(
            target=self.multiprocess_inference,
            args=(sys_prompt, user_prompt, max_tokens, *args),
            kwargs=kwargs,
        )
        p.start()
        p.join()

        # result_queue.get() function as a pop. Always save it to a variable or return it directly
        return result_queue.get()

    def load_model(self) -> Tuple[Qwen2VLForConditionalGeneration, AutoProcessor]:
        """
        Loads and returns the model to make inferences on
        """
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(self.model_name)
        return model, processor
