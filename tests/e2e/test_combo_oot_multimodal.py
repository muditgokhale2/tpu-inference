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

import pytest
import torch
from vllm.assets.image import ImageAsset
from vllm.model_executor.models.registry import ModelRegistry
from vllm.multimodal.image import convert_image_mode

# Official tpu_inference libraries and registries
from tpu_inference.models.common.model_loader import (_MODEL_REGISTRY,
                                                      register_model)
from tpu_inference.models.jax.qwen2_5_vl import \
    Qwen2_5_VLForConditionalGeneration
from vllm import LLM, EngineArgs, SamplingParams

try:
    from vllm.model_executor.models.interfaces_base import is_vllm_model
    VLLM_INTERFACE_CHECK_AVAILABLE = True
except ImportError:
    VLLM_INTERFACE_CHECK_AVAILABLE = False


# 1. Define OOT Class inheriting from the correct JAX class
class OOTMultimodalModel(Qwen2_5_VLForConditionalGeneration):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(
            f"!!! OOT Multimodal Model ({self.__class__.__name__}) Initialized !!!"
        )


# Standard gold-standard texts for accuracy check (aligned with test_multi_modal_inference.py)
EXPECTED_TEXTS = (
    "The image depicts a tall, cylindrical tower with a lattice-like structure, surrounded by cherry blossom trees in full bloom. The cherry blossoms are in various stages of opening, with pink petals covering the branches. The sky is clear and blue, providing a vibrant backdrop to the scene. The tower appears to be a significant landmark",
    "The image depicts a stunning view of the Tokyo Skytree, a tall broadcasting tower located in the Odaiba district of Tokyo, Japan. The skytree is surrounded by cherry blossom trees in full bloom, creating a picturesque and vibrant scene. The cherry blossoms are in various stages of bloom, with some branches densely covered",
)


def _get_tensor_parallel_size():
    return 2 if os.environ.get('TPU_VERSION', 'tpu6e') == "tpu7x" else 1


@pytest.fixture
def cleanup_registries():
    """Ensures a clean state for registries, matching original test_model_loader logic."""
    _MODEL_REGISTRY.clear()
    if hasattr(ModelRegistry, "models"):
        ModelRegistry.models.clear()
    yield
    _MODEL_REGISTRY.clear()
    if hasattr(ModelRegistry, "models"):
        ModelRegistry.models.clear()


# Mock config needed for vLLM resolution check
class MockModelConfig:

    def __init__(self, architectures):
        self.hf_config = self._MockHfConfig(architectures)
        self.model_impl = "flax_nnx"

    class _MockHfConfig:

        def __init__(self, architectures):
            self.architectures = architectures


def test_oot_multimodal_full_stack_verification(cleanup_registries):
    """
    Combines static plumbing verification with dynamic E2E multimodal inference.
    """
    arch = "OOTVisionModelForCausalLM"

    # --- PHASE 1: STATIC VERIFICATION ---
    register_model(arch, OOTMultimodalModel)
    assert arch in _MODEL_REGISTRY

    model_config = MockModelConfig(architectures=[arch])
    vllm_compatible_model, _ = ModelRegistry.resolve_model_cls(
        architectures=[arch], model_config=model_config)

    assert vllm_compatible_model is not None
    assert issubclass(vllm_compatible_model, torch.nn.Module)
    assert issubclass(vllm_compatible_model, OOTMultimodalModel)

    if VLLM_INTERFACE_CHECK_AVAILABLE:
        assert is_vllm_model(vllm_compatible_model)

    # --- PHASE 2: DYNAMIC VERIFICATION ---

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    tensor_parallel_size = _get_tensor_parallel_size()
    max_model_len = 4096
    gpu_memory_utilization = 0.5

    # Aligning exactly with the original multimodal test's EngineArgs
    engine_args = EngineArgs(
        model=model_id,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=1,
        hf_overrides={"architectures": [arch]},
        mm_processor_kwargs={
            "size": {
                "longest_edge": 1003520,
                "shortest_edge": 3136
            },
            "fps": 1,
        },
        limit_mm_per_prompt={"image": 1},
    )

    # Convert to dict and perform the same cleanup/modification as original test
    engine_kwargs = asdict(engine_args)
    if engine_kwargs.get("additional_config") is None:
        engine_kwargs["additional_config"] = {}

    # TPU stability: empty cudagraph_capture_sizes
    engine_kwargs["compilation_config"]["cudagraph_capture_sizes"] = []

    # Clean up None values for Pydantic validation
    pass_config = engine_kwargs["compilation_config"].get("pass_config") or {}
    pass_config = {k: v for k, v in pass_config.items() if v is not None}
    engine_kwargs["compilation_config"]["pass_config"] = pass_config

    # Initialize Engine
    llm = LLM(**engine_kwargs)

    # Identity Check: Inspect the model instance actually running on TPU
    model_instance = llm.llm_engine.model_executor.driver_worker.model_runner.model
    assert issubclass(type(model_instance), OOTMultimodalModel), \
        "Runtime engine is not using the OOT registered class!"

    # Using Qwen2.5-VL prompt template
    # NOTE: other models may be different
    image = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              "What is the content of this image?<|im_end|>\n"
              "<|im_start|>assistant\n")

    inputs = {"prompt": prompt, "multi_modal_data": {"image": image}}
    sampling_params = SamplingParams(temperature=0, max_tokens=64)

    outputs = llm.generate(inputs, sampling_params)
    generated_text = outputs[0].outputs[0].text.strip()
    print(f"OOT Model Response: {generated_text}")

    # Accuracy similarity check
    similarity_score = max(
        difflib.SequenceMatcher(None, generated_text, expected,
                                autojunk=False).ratio()
        for expected in EXPECTED_TEXTS)
    print(f"Similarity Score: {similarity_score:.4f}")
    assert similarity_score >= 0.85, f"Response quality failed! Output: {generated_text}"

    llm.llm_engine.engine_core.shutdown()
