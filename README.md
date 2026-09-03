# TensorRT Extension for Stable Diffusion (Blackwell & RTX 50-Series Edition)

This is a modernized fork of NVIDIA's TensorRT extension for Stable Diffusion WebUI (**tested and verified with [stable-diffusion-webui-reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge)** and Automatic1111 / SD.Next), updated for **NVIDIA Blackwell GPUs (GeForce RTX 5070 Ti, RTX 5080, RTX 5090)** with **Compute Capability 12.0 (`sm_120`)** and **TensorRT 10.8+**.

Supports Stable Diffusion 1.5, 2.1, SDXL, SDXL Turbo, and LCM.

---

## Key Improvements in this Edition
- **Tested with reForge**: Fully tested and verified with `stable-diffusion-webui-reForge`. Includes automated GPU weight synchronization for `ldm_patched` offloading and forced `dynamo=False` TorchScript export to prevent `FakeTensor` device mismatches on newer PyTorch versions.
- **Blackwell (RTX 50-series) Support**: Full support for `sm_120` (cc120) compute capability (RTX 5070 Ti, 5080, 5090).
- **TensorRT 10.8+ I/O Tensor API**: Replaced deprecated/removed legacy bindings API with modern TensorRT 10 I/O Tensor runtime.
- **Dynamic Timing Cache**: Automatically generates and saves new tactic caches for Blackwell (`timing_cache_*_cc120.cache`).
- **Python 3.10 – 3.12 Compatible**: Removed obsolete `protobuf==3.20.2` and ancient `tensorrt==9.0.1.post11.dev4` / CUDA 11 pins. Includes compatibility patch for newer `onnx >= 1.17` (`float32_to_bfloat16`).
- **No Browser Timeouts**: Streaming generator with live progress reporting keeps WebUI connections alive during long ONNX and TensorRT export jobs.
- **Robust Buffer Allocation**: Two-pass I/O tensor shape setting prevents race conditions and eliminates unnecessary GPU reallocations per denoising step.
- **Safe Headless/CPU Mode**: Extension imports gracefully even when running without an active GPU or in headless environments.

---

## Requirements for RTX 5070 Ti / Blackwell
1. **NVIDIA Driver**:
   - Linux: >= 570.86 (or newer, e.g. 575.xx / 580.xx)
   - Windows: >= 572.16 (or newer)
2. **CUDA Toolkit / PyTorch**:
   - PyTorch compiled with CUDA 12.8+ (required for `sm_120` kernel support).
3. **TensorRT**:
   - TensorRT >= 10.8.0 (`tensorrt>=10.8.0` or `tensorrt-cu12>=10.8.0`), which includes native Blackwell `sm_120` kernels.

---

## Installation

### Automatic Installation (WebUI / reForge)
1. Start your Stable Diffusion WebUI or reForge.
2. Go to the **Extensions** tab -> **Install from URL**.
3. Paste the URL:
   ```
   https://github.com/ventus34/Stable-Diffusion-WebUI-TensorRT-Blackwell.git
   ```
4. Click **Install**.
5. Restart the WebUI completely. During startup, `install.py` will verify or install the necessary TensorRT 10.8+ packages.

### Manual / Pre-installation
If you wish to install dependencies in your WebUI / reForge venv ahead of time:
```bash
# In your WebUI / reForge python environment:
pip install "tensorrt>=10.8.0" polygraphy onnx-graphsurgeon onnx onnxscript optimum --extra-index-url https://pypi.nvidia.com
```

---

## How to use

1. Select your target checkpoint (e.g. SD 1.5 or SDXL) in WebUI.
2. Navigate to the **TensorRT** tab.
3. Choose a preset:
   - For SD 1.5/2.1: Default covers 512x512 to 768x768 (batch size 1-4).
   - For SDXL: Default covers 1024x1024 (batch size 1).
4. Click **Export Default Engine** (or configure custom dimensions in Advanced Settings and click Export).
   - *Note: The first build will take 2-8 minutes as TensorRT profiles kernels and compiles the engine for your RTX 5070 Ti.*
5. In WebUI Settings -> **User Interface** -> **Quicksettings list**, add `sd_unet`. Apply settings and reload UI.
6. Select the compiled `[TRT] ...` engine from the `sd_unet` dropdown (or leave it on `Automatic`).
7. Generate images at maximum TensorRT speed!

### LoRA
To use LoRA checkpoints with TensorRT:
1. Go to the **TensorRT** -> **TensorRT LoRA** tab.
2. Select your LoRA checkpoint from the dropdown.
3. Click **Convert to TensorRT** (this takes ~15-20 seconds).
4. Use the LoRA normally in your prompt (e.g. `<lora:my_lora:1.0>`).

---

## Troubleshooting & Tips
- **Hires.fix**: Ensure the compiled dynamic engine covers both your base resolution (e.g. 512x512) and your upscaled resolution (e.g. 1024x1024).
- **Resolution**: Both height and width must be divisible by 64.
- **Timing Cache**: Compiled tactic cache will be saved under `timing_caches/timing_cache_<os>_cc120.cache`, speeding up subsequent engine builds.
