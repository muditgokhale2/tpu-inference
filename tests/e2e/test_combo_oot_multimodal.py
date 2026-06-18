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
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLDummyInputsBuilder, Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo)
from vllm.model_executor.models.registry import ModelRegistry
# Import official multimodal registration tools
from vllm.multimodal import MULTIMODAL_REGISTRY
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

# --- MODULE LEVEL REGISTRATION ---
custom_arch = "My_Inherited_OOT_Multimodal_Model"


# 1. Define OOT Class with Decorator (Sets up the JAX class identity)
@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder,
)
class OOTMultimodalModel(Qwen2_5_VLForConditionalGeneration):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Provenance signature to verify active process execution
        print(
            f"!!! OOT Multimodal Subclass ({self.__class__.__name__}) Successfully Initialized !!!"
        )


# 2. Register via official tpu_inference mechanism (creates shadow class)
register_model(custom_arch, OOTMultimodalModel)


# 3. Reference implementation of MockModelConfig (from test_model_loader.py)
# This is used to resolve the shadow class from vLLM's registry.
class MockModelConfig:

    def __init__(self, architectures):
        self.hf_config = self._MockHfConfig(architectures)
        self.model_impl = "flax_nnx"

    class _MockHfConfig:

        def __init__(self, architectures):
            self.architectures = architectures


# Extract the shadow class that was registered in step 2.
vllm_compatible_model, _ = ModelRegistry.resolve_model_cls(
    architectures=[custom_arch],
    model_config=MockModelConfig(architectures=[custom_arch]))

# 4. CRITICAL SYNC: Manually copy the MM factory to the shadow class.
# This ensures that ModelConfig.is_multimodal_model will be True in any process.
has_factory = hasattr(OOTMultimodalModel, "_processor_factory")
print(f"DEBUG SYNC [PID {os.getpid()}] | JAX class has factory: {has_factory}")

if has_factory:
    vllm_compatible_model._processor_factory = OOTMultimodalModel._processor_factory
    print(
        f"DEBUG SYNC [PID {os.getpid()}] | Successfully injected factory into shadow class."
    )
else:
    print(
        f"DEBUG SYNC [PID {os.getpid()}] | WARNING: Factory missing on JAX class!"
    )

# Standard gold-standard texts for accuracy check
EXPECTED_TEXTS = (
    "The image depicts a tall, cylindrical tower with a lattice-like structure, surrounded by cherry blossom trees in full bloom. The cherry blossoms are in various stages of opening, with pink petals covering the branches. The sky is clear and blue, providing a vibrant backdrop to the scene. The tower appears to be a significant landmark",
    "The image depicts a stunning view of the Tokyo Skytree, a tall broadcasting tower located in the Odaiba district of Tokyo, Japan. The skytree is surrounded by cherry blossom trees in full bloom, creating a picturesque and vibrant scene. The cherry blossoms are in various stages of bloom, with some branches densely covered",
)


def _get_tensor_parallel_size():
    return 2 if os.environ.get('TPU_VERSION', 'tpu6e') == "tpu7x" else 1


def test_oot_multimodal_full_stack_verification():
    """
    Combined Feature Test: OOT Inheritance + Multimodal Inference.
    
    This follows the registration pattern in test_model_loader.py while 
    extending it to support multimodal processor synchronization.
    """
    # Verify core registration properties
    assert custom_arch in _MODEL_REGISTRY
    assert vllm_compatible_model is not None
    assert issubclass(vllm_compatible_model, OOTMultimodalModel)

    if VLLM_INTERFACE_CHECK_AVAILABLE:
        assert is_vllm_model(vllm_compatible_model)

    # --- DYNAMIC VERIFICATION ---
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    engine_args = EngineArgs(
        model=model_id,
        # Force vLLM to resolve to our new custom architecture name
        hf_overrides={"architectures": [custom_arch]},
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
    llm = LLM(**engine_kwargs)

    # Identity Check: Verify the model runner is using our OOT class instance
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
