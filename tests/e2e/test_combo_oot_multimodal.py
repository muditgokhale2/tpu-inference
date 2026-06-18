# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import difflib
import os
from dataclasses import asdict

from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode

# Official tpu_inference libraries and registries
from tpu_inference.models.common.model_loader import _MODEL_REGISTRY
from tpu_inference.models.jax.qwen2_5_vl import \
    Qwen2_5_VLForConditionalGeneration
from vllm import LLM, EngineArgs, SamplingParams

# try:
#     from vllm.model_executor.models.interfaces_base import is_vllm_model
#     VLLM_INTERFACE_CHECK_AVAILABLE = True
# except ImportError:
#     VLLM_INTERFACE_CHECK_AVAILABLE = False

# --- MODULE LEVEL "SUBSTITUTION" ---
# Instead of creating a new architecture name, we override the JAX implementation
# of the official architecture. This allows us to "inherit" all of vLLM's
# built-in multimodal registration and identity checks for the official name.

official_arch = "Qwen2_5_VLForConditionalGeneration"


# 1. Define OOT Class
class OOTMultimodalModel(Qwen2_5_VLForConditionalGeneration):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Provenance signature to prove this class is active
        print(
            f"!!! OOT Multimodal Subclass ({self.__class__.__name__}) Successfully Intercepted !!!"
        )


# 2. Substitution: Overwrite the tpu_inference registry for the official name.
# When tpu_runner.py calls get_model(), it will look up this dictionary and
# find our OOT class instead of the standard JAX implementation.
_MODEL_REGISTRY[official_arch] = OOTMultimodalModel

# Standard gold-standard texts for accuracy check
EXPECTED_TEXTS = (
    "The image depicts a tall, cylindrical tower with a lattice-like structure, surrounded by cherry blossom trees in full bloom. The cherry blossoms are in various stages of opening, with pink petals covering the branches. The sky is clear and blue, providing a vibrant backdrop to the scene. The tower appears to be a significant landmark",
    "The image depicts a stunning view of the Tokyo Skytree, a tall broadcasting tower located in the Odaiba district of Tokyo, Japan. The skytree is surrounded by cherry blossom trees in full bloom, creating a picturesque and vibrant scene. The cherry blossoms are in various stages of bloom, with some branches densely covered",
)


def _get_tensor_parallel_size():
    return 2 if os.environ.get('TPU_VERSION', 'tpu6e') == "tpu7x" else 1


def test_oot_multimodal_full_stack_verification():
    """
    Combined Feature Test: OOT Inheritance via Substitution.
    
    This verifies that by overwriting the official architecture entry in 
    _MODEL_REGISTRY, we can run custom logic while maintaining full 
    parity with official multimodal configurations.
    """
    # Verify substitution is active in the current process
    assert _MODEL_REGISTRY[official_arch] is OOTMultimodalModel

    # --- DYNAMIC VERIFICATION ---
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    engine_args = EngineArgs(
        model=model_id,
        # NO hf_overrides needed! We are pretending to be the official model.
        max_model_len=4096,
        tensor_parallel_size=_get_tensor_parallel_size(),
        gpu_memory_utilization=0.5,
        max_num_seqs=1,
        mm_processor_kwargs={
            "size": {
                "longest_edge": 1003520,
                "shortest_edge": 3136
            },
            "fps": 1,
        },
        limit_mm_per_prompt={"image": 1},
    )

    engine_kwargs = asdict(engine_args)
    if engine_kwargs.get("additional_config") is None:
        engine_kwargs["additional_config"] = {}
    engine_kwargs["compilation_config"]["cudagraph_capture_sizes"] = []

    pass_config = engine_kwargs["compilation_config"].get("pass_config") or {}
    pass_config = {k: v for k, v in pass_config.items() if v is not None}
    engine_kwargs["compilation_config"]["pass_config"] = pass_config

    # Initialize Engine.
    # Because we use the official name, vLLM finds its own registered MM labels.
    llm = LLM(**engine_kwargs)

    # Identity Check: Verify we are using the subclass, not the original class
    # The runner's model will be an instance of OOTMultimodalModel because of our _MODEL_REGISTRY swap.
    model_instance = llm.llm_engine.model_executor.driver_worker.model_runner.model
    assert isinstance(model_instance, OOTMultimodalModel)

    # Inference Quality Check
    image = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              "What is the content of this image?<|im_end|>\n"
              "<|im_start|>assistant\n")

    inputs = {"prompt": prompt, "multi_modal_data": {"image": image}}
    outputs = llm.generate(inputs, SamplingParams(temperature=0,
                                                  max_tokens=64))
    generated_text = outputs[0].outputs[0].text.strip()
    print(f"\nOOT Subclass Verified Response: {generated_text}")

    # Accuracy similarity check
    similarity_score = max(
        difflib.SequenceMatcher(None, generated_text, expected,
                                autojunk=False).ratio()
        for expected in EXPECTED_TEXTS)
    print(f"Similarity Score: {similarity_score:.4f}")
    assert similarity_score >= 0.85

    llm.llm_engine.engine_core.shutdown()
