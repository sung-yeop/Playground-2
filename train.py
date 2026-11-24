"""Fine-tune Qwen with one CUDA GPU. Input JSONL has prompt and answer strings."""

import argparse
import json
import math
from pathlib import Path
from uuid import uuid4

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training
from risorius_rl import RecordingSession
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed


MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def read_examples(path, tokenizer, max_length):
    examples = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("prompt", "answer")):
                raise ValueError(f"{path}:{line_number} needs nonempty prompt and answer strings")
            messages = [{"role": "user", "content": row["prompt"]}]
            prefix = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            messages.append({"role": "assistant", "content": row["answer"]})
            tokens = tokenizer.apply_chat_template(messages)
            if tokens[:len(prefix)] != prefix or len(prefix) >= len(tokens):
                raise ValueError(f"{path}:{line_number} has an incompatible chat template")
            if len(tokens) > max_length:
                raise ValueError(f"{path}:{line_number} exceeds {max_length} tokens")
            labels = [-100] * len(prefix) + tokens[len(prefix):]
            examples.append((torch.tensor([tokens]), torch.tensor([labels])))
    if not examples:
        raise ValueError(f"{path} contains no examples")
    return examples


def loss_for(model, example):
    tokens, labels = example
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=model.device.type == "cuda"):
        return model(input_ids=tokens.to(model.device), labels=labels.to(model.device)).loss


@torch.no_grad()
def evaluate(model, examples):
    model.eval()
    total_loss = 0.0
    token_count = 0
    for example in examples:
        count = (example[1][:, 1:] != -100).sum().item()
        total_loss += loss_for(model, example).item() * count
        token_count += count
    return total_loss / token_count


def fit(model, training, evaluation, steps, recording):
    def log(metrics, step, phase):
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"Nonfinite {phase} metric at step {step}")
        print(json.dumps({"phase": phase, "step": step, **metrics}), flush=True)
        recording.log_metrics(metrics, step=step, phase=phase)

    log({"loss": evaluate(model, evaluation)}, 0, "eval")
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    for step in range(1, steps + 1):
        loss = loss_for(model, training[(step - 1) % len(training)])
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        metrics = {"loss": value}
        if model.device.type == "cuda":
            metrics["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        log(metrics, step, "train")
    log({"loss": evaluate(model, evaluation)}, steps, "eval")


def save_adapter(model, tokenizer, directory):
    # Write only the loadable adapter and tokenizer files.
    config = model.peft_config["default"]
    config.inference_mode = True
    config.save_pretrained(directory)
    weights = get_peft_model_state_dict(model, save_embedding_layers=False)
    save_file({k: v.cpu().contiguous() for k, v in weights.items()},
              str(directory / "adapter_model.safetensors"), metadata={"format": "pt"})
    tokenizer.save_pretrained(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-data", type=Path, default=Path("examples/train.jsonl"))
    parser.add_argument("--eval-data", type=Path, default=Path("examples/eval.jsonl"))
    parser.add_argument("--riso-config", type=Path, help="Load Risorius recording settings from YAML")
    args = parser.parse_args()
    if args.steps < 1 or args.max_length < 2:
        parser.error("steps must be positive and max-length must be at least 2")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("training requires a BF16-capable CUDA GPU")

    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    training = read_examples(args.train_data, tokenizer, args.max_length)
    evaluation = read_examples(args.eval_data, tokenizer, args.max_length)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, quantization_config=quantization,
        dtype=torch.bfloat16, device_map={"": 0},
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, gradient_checkpointing_kwargs={"use_reentrant": False})
    config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, target_modules="all-linear")
    model = get_peft_model(base, config, revision=REVISION)
    model.print_trainable_parameters()

    run = Path("outputs") / uuid4().hex
    if args.riso_config is None:
        session = RecordingSession(project="playground-4090", directory=str(run / "records"), upload=False)
    else:
        session = RecordingSession.from_config(args.riso_config)
        if session.directory is None:
            parser.error(f"Could not initialize Risorius recording from {args.riso_config}")
    with session as recording:
        print(f"Recording directory: {recording.directory}", flush=True)
        fit(model, training, evaluation, args.steps, recording)
    save_adapter(model, tokenizer, run / "adapter")
    print(f"Adapter: {run / 'adapter'}")
    if session.upload:
        print(f"Delivery status: {session.directory / 'delivery-status.json'}")


if __name__ == "__main__":
    main()
