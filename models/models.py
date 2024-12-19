import gc
from abc import ABC
from multiprocessing import Process, Queue
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

#######
# INTERFACES
#######


class ModelInterface(LLM, ABC):
    model_name: str = None
    capabilities: List[str] = []
    skip_special_tokens: bool = False
    max_token: int = 10000
    temperature: float = 0.01
    top_p: float = 0.9
    history_len: int = 0

    @property
    def _llm_type(self) -> str:
        return self.__class__.__name__

    @property
    def _history_len(self) -> int:
        return self.history_len

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_token": self.max_token,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "history_len": self.history_len,
        }

    def _call(
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


class QwenVLModel(ModelInterface):
    capabilities: List[str] = ["image", "text"]

    def __init__(
        self,
        model_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model_name: str = model_name

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def inference(
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

    def _call(
        self,
        sys_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
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
            target=self.inference,
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
