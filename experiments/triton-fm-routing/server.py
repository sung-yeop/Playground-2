import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

models = json.loads(Path("models.json").read_text())
text_path = models["text"]["path"]
tokenizer = AutoTokenizer.from_pretrained(text_path)
language = (
    AutoModelForCausalLM.from_pretrained(text_path, dtype=torch.bfloat16)
    .to("cuda")
    .eval()
)
vision = AutoModel.from_pretrained(models["image"]["path"]).to("cuda").eval()


@batch
def text_infer(prompt):
    results = []
    with torch.inference_mode():
        for item in prompt[:, 0]:
            tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": item.decode()}],
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to("cuda")
            generated = language.generate(
                tokens,
                attention_mask=torch.ones_like(tokens),
                max_new_tokens=24,
                do_sample=False,
            )
            answer = tokenizer.decode(
                generated[0, tokens.shape[1] :], skip_special_tokens=True
            )
            results.append([answer.strip().encode()])
    return {"answer": np.array(results, dtype=object)}


@batch
def image_infer(pixels):
    with torch.inference_mode():
        x = (
            torch.as_tensor(pixels.copy(), device="cuda").permute(0, 3, 1, 2).float()
            / 255
        )
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        features = vision(pixel_values=(x - mean) / std).last_hidden_state[:, 0]
        return {"embedding": features.contiguous().cpu().numpy()}


config = TritonConfig(
    http_address="127.0.0.1",
    grpc_address="127.0.0.1",
    metrics_address="127.0.0.1",
    http_port=18400,
    grpc_port=18401,
    metrics_port=18402,
)
with Triton(config=config) as triton:
    triton.bind(
        model_name="text_fm",
        infer_func=text_infer,
        inputs=[Tensor(name="prompt", dtype=np.bytes_, shape=(1,))],
        outputs=[Tensor(name="answer", dtype=np.bytes_, shape=(1,))],
        config=ModelConfig(max_batch_size=4),
    )
    triton.bind(
        model_name="image_fm",
        infer_func=image_infer,
        inputs=[Tensor(name="pixels", dtype=np.uint8, shape=(224, 224, 3))],
        outputs=[Tensor(name="embedding", dtype=np.float32, shape=(384,))],
        config=ModelConfig(max_batch_size=4),
    )
    triton.serve()
