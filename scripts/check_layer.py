"""Checks VK_LAYER_stage_profiler against the app's own measurements (Phase 7).

Usage: python3 scripts/check_layer.py PATH_TO_gpu_profiler LAYER_DIR [--vkcube [VKCUBE]]

VKCUBE defaults to "vkcube". On macOS, Homebrew's vkcube is a wrapper that
launches an app bundle without passing the environment through, so pass the
real binary, e.g. .../vulkan-tools/<version>/cube/vkcube.app/Contents/MacOS/vkcube

1. Pass-through: the app still runs and prints its report with the layer on.
2. Command counts: the layer's dispatch/barrier/copy counts equal the app's.
3. Agreement: the layer's mean fence wait is within 10% of the app's.
4. Stacked: with the validation layer also on, there are 0 validation messages.
5. With --vkcube: the layer reports submits for vkcube, an unmodified app.
"""

import os
import re
import subprocess
import sys

app, layer_dir = sys.argv[1], sys.argv[2]
env = dict(os.environ, VK_ADD_LAYER_PATH=layer_dir,
           VK_LOADER_LAYERS_ENABLE="VK_LAYER_stage_profiler")
failures = []


def run(args, use_layer=True):
    result = subprocess.run(args, env=env if use_layer else os.environ, capture_output=True,
                            text=True, timeout=600)
    return result.returncode, result.stdout + result.stderr


def number(pattern, text, group=1):
    match = re.search(pattern, text)
    if match is None:
        failures.append(f"missing output matching {pattern!r}")
        return None
    return float(match.group(group))


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        failures.append(name)


report = [app, "--report", "--n", "2048", "--no-validation"]

# 1. Pass-through.
code_without, without = run(report, use_layer=False)
code_with, out = run(report)
check("pass-through", code_with == 0 and "frame" in out and "[stage_profiler]" in out,
      f"exit {code_with}, report printed, layer report present")

# 2. Command counts.
for kind in ("dispatch", "barrier", "copy"):
    app_count = number(rf"app recorded commands:.*{kind}=(\d+)", out)
    layer_count = number(rf"\[stage_profiler\] recorded commands:.*{kind}=(\d+)", out)
    check(f"{kind} count", app_count == layer_count, f"app {app_count}, layer {layer_count}")
app_waits = number(r"app fence waits: count=(\d+)", out)
layer_waits = number(r"vkWaitForFences\s+calls=(\d+)", out)
check("fence wait count", app_waits == layer_waits, f"app {app_waits}, layer {layer_waits}")

# 3. Agreement on mean wait time.
app_mean = number(r"app fence waits:.*mean_us=([\d.]+)", out)
layer_mean = number(r"vkWaitForFences\s+calls=\d+ total_us=[\d.]+ mean_us=([\d.]+)", out)
if app_mean and layer_mean is not None:
    diff = abs(layer_mean - app_mean) / app_mean
    check("wait agreement", diff < 0.10,
          f"app {app_mean:.1f} µs, layer {layer_mean:.1f} µs per wait, differ by {diff:.2%}")

# 4. Stacked with the validation layer.
code, out = run([app, "--report", "--n", "512", "--steps", "40", "--warmup", "4",
                 "--batch", "4"])
messages = len(re.findall(r"\[vulkan (warning|error)\]", out))
check("stacked with validation", code == 0 and messages == 0 and "[stage_profiler]" in out,
      f"exit {code}, {messages} validation messages")

# 5. An unmodified app.
if "--vkcube" in sys.argv:
    at = sys.argv.index("--vkcube")
    vkcube = sys.argv[at + 1] if at + 1 < len(sys.argv) else "vkcube"
    code, out = run([vkcube, "--c", "300"])
    submits = number(r"vkQueueSubmit\s+calls=(\d+)", out)
    check("vkcube", code == 0 and submits is not None and submits > 0,
          f"exit {code}, {submits} vkQueueSubmit calls")
    print("\n".join(line for line in out.splitlines() if "[stage_profiler]" in line))

sys.exit(1 if failures else 0)
