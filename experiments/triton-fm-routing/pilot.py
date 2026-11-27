import hashlib
import json
from pathlib import Path
import numpy as np
from risorius_rl import Run
from tritonclient import http

models = json.loads(Path("models.json").read_text())
client = http.InferenceServerClient("127.0.0.1:18400")
tasks = [
    (
        "Write the word triton in all capital letters. Respond with the word alone.",
        "text_fm",
    ),
    ("Extract the feature vector of the attached synthetic RGB image.", "image_fm"),
]


def infer(model, name, datatype, value, output):
    item = http.InferInput(name, value.shape, datatype)
    item.set_data_from_numpy(value)
    return client.infer(model, [item]).as_numpy(output)


def text(prompt):
    value = np.array([[prompt.encode()]], dtype=object)
    return infer("text_fm", "prompt", "BYTES", value, "answer")[0, 0].decode().strip()


def execute(selected, task):
    if selected == "text_fm":
        return {"answer": text(task)}
    pixels = (np.arange(224 * 224 * 3) % 251).astype(np.uint8).reshape(1, 224, 224, 3)
    embedding = infer("image_fm", "pixels", "UINT8", pixels, "embedding")
    if embedding.shape != (1, 384) or not np.isfinite(embedding).all():
        raise ValueError("Unexpected vision embedding")
    return {
        "dimensions": int(embedding.shape[1]),
        "l2_norm": float(np.linalg.norm(embedding)),
        "sha256": hashlib.sha256(embedding.tobytes()).hexdigest(),
    }


with Run() as run:
    records = []
    for task, expected in tasks:
        with run.span(
            "swarm.task",
            kind="tool",
            phase="eval",
            inputs={"task": task},
            attributes={"session_id": run.run_id},
        ) as root:
            with run.span(
                "router.select",
                kind="llm",
                phase="eval",
                inputs=task,
                attributes={
                    "model": models["text"]["repo"],
                    "revision": models["text"]["revision"],
                },
            ) as decision:
                selected = text(
                    "Classify the following quoted task by modality.\nTask: "
                    + json.dumps(task)
                    + "\nWhich tool is appropriate for this task? Choose text_fm (language) or image_fm (visual features). Reply with the tool name only."
                )
                decision.set_output(selected)
            if selected not in {"text_fm", "image_fm"}:
                raise ValueError("Invalid tool selection: " + selected)
            model = models["text" if selected == "text_fm" else "image"]
            with run.span(
                "executor." + selected,
                kind="model",
                phase="eval",
                inputs=task,
                attributes={"model": model["repo"], "revision": model["revision"]},
            ) as call:
                output = execute(selected, task)
                call.set_output(output)
            correct = selected == expected and (
                output.get("answer") == "TRITON"
                if expected == "text_fm"
                else output.get("l2_norm", 0) > 0
            )
            root.set_output({"selected": selected, "passed": correct, "result": output})
            records.append(
                {
                    "task": task,
                    "root": str(root),
                    "model": model["repo"],
                    "revision": model["revision"],
                    "selected": selected,
                    "passed": correct,
                    "result": output,
                }
            )
    passed = sum(record["passed"] for record in records)
    evidence = Path("evidence-" + run.run_id + ".json")
    evidence.write_text(json.dumps(records, indent=2))
    run.log_metrics({"tasks": len(tasks), "passed": passed}, step=2, phase="eval")
    run.upload(evidence, name="pilot-evidence.json")
    if passed != len(tasks):
        raise RuntimeError("Task verification failed: " + evidence.read_text())
    run.finish(metrics={"tasks_passed": passed})
    print("FM_PILOT_PASSED", run.run_id, evidence.name)
