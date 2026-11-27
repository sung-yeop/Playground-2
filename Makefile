.PHONY: serve train smoke
serve train smoke:
	$(MAKE) -C experiments/qwen-lora-sft $@
