import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

import train


class PublishSnapshotTest(unittest.TestCase):
    def test_cpu_merge_bypasses_peft_auto_placement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_dir = root / "base"
            config = Qwen2Config(vocab_size=4, hidden_size=16, intermediate_size=32,
                                 num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)
            model = Qwen2ForCausalLM(config).to(torch.bfloat16)
            model.save_pretrained(base_dir)
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"x": 0})))
            adapter = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=2, target_modules=["q_proj"]))
            layer = adapter.base_model.model.model.layers[0].self_attn.q_proj
            with torch.no_grad():
                layer.lora_B["default"].weight.fill_(0.125)
                expected = layer.base_layer.weight + layer.get_delta_weight("default").to(torch.bfloat16)
            train.save_adapter(adapter, tokenizer, root / "adapter")

            # A GPU host can choose a mixed device map even for a CPU-loaded base.
            run = Mock()
            with (
                patch.object(train, "MODEL", str(base_dir)),
                patch("peft.peft_model.infer_auto_device_map", side_effect=AssertionError("automatic dispatch")),
                patch("peft.peft_model.infer_device", side_effect=AssertionError("automatic device selection")),
            ):
                train.publish_serving_snapshot(run, root, tokenizer)

            snapshot = root / "serving-model"
            restored = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.bfloat16)
            torch.testing.assert_close(restored.model.layers[0].self_attn.q_proj.weight, expected, rtol=0, atol=0)
            manifest = root / "snapshot.sha256"
            lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in sorted(snapshot.iterdir())]
            self.assertEqual(manifest.read_text().splitlines(), lines)
            run.upload.assert_called_once_with(manifest, checkpoint=True)
