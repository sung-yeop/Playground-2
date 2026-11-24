UV = uv run --python 3.12
SERVE_MODEL ?= Qwen/Qwen2.5-0.5B-Instruct
ifeq ($(SERVE_MODEL),Qwen/Qwen2.5-0.5B-Instruct)
SERVE_REVISION ?= 7ae557604adf67be50417f59c2c2f167def9a775
endif
STEPS ?= 8
TRAIN_DATA ?= examples/train.jsonl
EVAL_DATA ?= examples/eval.jsonl

.PHONY: serve train smoke
# Run on the serving host with its own CUDA/PyTorch environment.
serve:
	$(UV) --no-project --isolated --with vllm==0.18.0 vllm serve "$(SERVE_MODEL)" \
	  $(if $(SERVE_REVISION),--revision "$(SERVE_REVISION)") --served-model-name playground \
	  --host 127.0.0.1 --port 8000 --dtype bfloat16 --gpu-memory-utilization 0.80 \
	  --max-model-len 1024 --max-num-seqs 2 --max-num-batched-tokens 1024 --enforce-eager

# Run on the training host. Smoke exercises one complete optimizer step.
smoke: STEPS = 1
train smoke:
	$(UV) --frozen python train.py --steps "$(STEPS)" --train-data "$(TRAIN_DATA)" \
	  --eval-data "$(EVAL_DATA)"
