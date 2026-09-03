import os
import sys
import math
import time
import shutil
import json
from pathlib import Path
from logging import info, error
from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import onnx
from onnx import numpy_helper
try:
    from optimum.onnx.utils import (
        _get_onnx_external_data_tensors,
        check_model_uses_external_data,
    )
except ImportError:
    def check_model_uses_external_data(model):
        for initializer in model.graph.initializer:
            if getattr(initializer, "data_location", None) == onnx.TensorProto.EXTERNAL:
                return True
        return False

    def _get_onnx_external_data_tensors(model):
        paths = []
        for initializer in model.graph.initializer:
            if getattr(initializer, "data_location", None) == onnx.TensorProto.EXTERNAL:
                for entry in initializer.external_data:
                    if entry.key == "location":
                        paths.append(entry.value)
        return paths


from modules import shared

from utilities import Engine
from datastructures import ProfileSettings
from model_helper import UNetModel


def apply_lora(model: torch.nn.Module, lora_path: str, inputs: Tuple[torch.Tensor]) -> torch.nn.Module:
    try:
        import sys

        sys.path.append("extensions-builtin/Lora")
        import importlib

        networks = importlib.import_module("networks")
        network = importlib.import_module("network")
        lora_net = importlib.import_module("extra_networks_lora")
    except Exception as e:
        error(e)
        error("LoRA not found. Please install LoRA extension first from ...")
    model.forward(*inputs)
    lora_name = os.path.splitext(os.path.basename(lora_path))[0]
    networks.load_networks(
        [lora_name], [1.0], [1.0], [None]
    )

    model.forward(*inputs)
    return model


def get_refit_weights(
    state_dict: dict, onnx_opt_path: str, weight_name_mapping: dict, weight_shape_mapping: dict
) -> dict:
    refit_weights = OrderedDict()
    onnx_opt_dir = os.path.dirname(onnx_opt_path)
    onnx_opt_model = onnx.load(onnx_opt_path)
    # Create initializer data hashes
    initializer_hash_mapping = {}
    onnx_data_mapping = {}
    for initializer in onnx_opt_model.graph.initializer:
        initializer_data = numpy_helper.to_array(
            initializer, base_dir=onnx_opt_dir
        ).astype(np.float16)
        initializer_hash = hash(initializer_data.data.tobytes())
        initializer_hash_mapping[initializer.name] = initializer_hash
        onnx_data_mapping[initializer.name] = initializer_data

    for torch_name, initializer_name in weight_name_mapping.items():
        initializer_hash = initializer_hash_mapping[initializer_name]
        wt = state_dict[torch_name]

        # get shape transform info
        initializer_shape, is_transpose = weight_shape_mapping[torch_name]
        if is_transpose:
            wt = torch.transpose(wt, 0, 1)
        else:
            wt = torch.reshape(wt, initializer_shape)

        # include weight if hashes differ
        wt_hash = hash(wt.cpu().detach().numpy().astype(np.float16).data.tobytes())
        if initializer_hash != wt_hash:
            delta = wt - torch.tensor(onnx_data_mapping[initializer_name]).to(wt.device)
            refit_weights[initializer_name] = delta.contiguous()

    return refit_weights


def export_lora(
    modelobj: UNetModel,
    onnx_path: str,
    weights_map_path: str,
    lora_name: str,
    profile: ProfileSettings,
) -> dict:
    info("Exporting to ONNX...")
    inputs = modelobj.get_sample_input(
        profile.bs_opt * 2,
        profile.h_opt // 8,
        profile.w_opt // 8,
        profile.t_opt,
    )

    with open(weights_map_path, "r") as fp_wts:
        print(f"[I] Loading weights map: {weights_map_path} ")
        [weights_name_mapping, weights_shape_mapping] = json.load(fp_wts)

    with torch.inference_mode(), torch.autocast("cuda"):
        modelobj.unet = apply_lora(
            modelobj.unet, os.path.splitext(lora_name)[0], inputs
        )

        refit_dict = get_refit_weights(
            modelobj.unet.state_dict(),
            onnx_path,
            weights_name_mapping,
            weights_shape_mapping,
        )

    return refit_dict


def _math_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
):
    scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    if is_causal:
        L, S = query.size(-2), key.size(-2)
        mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        scores = scores.masked_fill(~mask, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, value)


def swap_sdpa(func):
    def wrapper(*args, **kwargs):
        old_sdpa = getattr(F, "scaled_dot_product_attention", None)
        # Instead of deleting scaled_dot_product_attention (which causes AttributeError in reForge/ldm_patched),
        # replace it with a pure-math implementation that is 100% exportable to ONNX across all backends.
        try:
            setattr(F, "scaled_dot_product_attention", _math_scaled_dot_product_attention)
        except Exception:
            pass
        try:
            ret = func(*args, **kwargs)
        finally:
            if old_sdpa is not None:
                try:
                    setattr(F, "scaled_dot_product_attention", old_sdpa)
                except Exception:
                    pass
        return ret

    return wrapper


@swap_sdpa
def export_onnx(
    onnx_path: str,
    modelobj: UNetModel,
    profile: ProfileSettings,
    opset: int = 17,
    diable_optimizations: bool = False,
):
    print(f"[TensorRT] Przygotowywanie wag do eksportu ONNX...", flush=True)
    inputs = modelobj.get_sample_input(
        profile.bs_opt * 2,
        profile.h_opt // 8,
        profile.w_opt // 8,
        profile.t_opt,
    )

    if not os.path.exists(onnx_path):
        _export_onnx(
            modelobj.unet,
            inputs,
            Path(onnx_path),
            opset,
            modelobj.get_input_names(),
            modelobj.get_output_names(),
            modelobj.get_dynamic_axes(),
            modelobj.optimize if not diable_optimizations else None,
        )
    else:
        print(f"[TensorRT] Znaleziono istniejący plik ONNX: {onnx_path} (pomijanie powtórnego eksportu)", flush=True)


def _export_onnx(
    model: torch.nn.Module, inputs: Tuple[torch.Tensor], path: str, opset: int, in_names: List[str], out_names: List[str], dyn_axes: dict, optimizer=None
):
    try:
        import onnxscript
    except ImportError:
        try:
            print("[TensorRT] Wykryto brak pakietu 'onnxscript' wymaganego przez nowsze wersje PyTorch. Instalowanie w tle...", flush=True)
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxscript"])
            import onnxscript
            print("[TensorRT] Pakiet 'onnxscript' został pomyślnie zainstalowany!", flush=True)
        except Exception as err:
            print(f"[TensorRT] Uwaga: Automatyczna instalacja onnxscript nie powiodła się: {err}", flush=True)

    tmp_dir = os.path.abspath("onnx_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, "model.onnx")
    # Ensure all model parameters and buffers are placed on the input device
    target_device = inputs[0].device if len(inputs) > 0 else torch.device("cuda")
    try:
        model = model.to(target_device)
        for p in model.parameters():
            if p.device != target_device:
                p.data = p.data.to(target_device)
        for b in model.buffers():
            if b.device != target_device:
                b.data = b.data.to(target_device)
    except Exception as e:
        print(f"[TensorRT] Info: Weryfikacja urządzeń parametrów modelu: {e}", flush=True)

    export_kwargs = {
        "export_params": True,
        "opset_version": max(opset, 17),
        "do_constant_folding": True,
        "input_names": in_names,
        "output_names": out_names,
        "dynamic_axes": dyn_axes,
    }

    # In PyTorch >= 2.2 / 2.4, explicitly set dynamo=False to use TorchScript tracer instead of TorchDynamo/FakeTensor
    import inspect
    sig = inspect.signature(torch.onnx.export)
    if "dynamo" in sig.parameters:
        export_kwargs["dynamo"] = False

    try:
        print(f"[TensorRT] Eksportowanie grafu PyTorch do ONNX (plik roboczy: {tmp_path})...", flush=True)
        print(f"[TensorRT] To może potrwać 1–3 minuty. Proszę czekać...", flush=True)
        with torch.inference_mode(), torch.autocast("cuda"):
            torch.onnx.export(
                model,
                inputs,
                tmp_path,
                **export_kwargs,
            )
        print(f"[TensorRT] Graf ONNX wyeksportowany pomyślnie!", flush=True)
    except Exception as e:
        print(f"[TensorRT BŁĄD] Eksport do ONNX nie powiódł się: {e}", flush=True)
        import traceback
        traceback.print_exc()
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"Błąd podczas torch.onnx.export: {e}")

    print(f"[TensorRT] Przetwarzanie i weryfikacja struktury ONNX...", flush=True)
    os.makedirs(path.parent, exist_ok=True)
    onnx_model = onnx.load(tmp_path, load_external_data=False)
    model_uses_external_data = check_model_uses_external_data(onnx_model)

    if model_uses_external_data:
        print(f"[TensorRT] Model przekracza 2GB (używa zewnętrznych danych wag). Zapisywanie...", flush=True)
        tensors_paths = _get_onnx_external_data_tensors(onnx_model)
        onnx_model = onnx.load(tmp_path, load_external_data=True)
        onnx.save(
            onnx_model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=path.name + "_data",
            size_threshold=1024,
        )

    if optimizer is not None:
        try:
            print(f"[TensorRT] Optymalizacja węzłów ONNX (GraphSurgeon)...", flush=True)
            onnx_opt_graph = optimizer("unet", onnx_model)
            onnx.save(onnx_opt_graph, path)
        except Exception as e:
            print(f"[TensorRT BŁĄD] Optymalizacja ONNX nie powiodła się: {e}", flush=True)
            import traceback
            traceback.print_exc()
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(f"Błąd podczas optymalizacji grafu ONNX: {e}")

    if not model_uses_external_data and optimizer is None:
        shutil.move(tmp_path, str(path))

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[TensorRT] Plik ONNX gotowy pod adresem: {path}", flush=True)


def export_trt(trt_path: str, onnx_path: str, timing_cache: str, profile: dict, use_fp16: bool):
    print(f"[TensorRT] Rozpoczynanie kompilacji silnika TensorRT...", flush=True)
    print(f"[TensorRT] Wejście ONNX: {onnx_path}", flush=True)
    print(f"[TensorRT] Wyjście TRT:  {trt_path}", flush=True)
    print(f"[TensorRT] Pamięć podręczna taktyk (timing cache): {timing_cache}", flush=True)
    engine = Engine(trt_path)

    model = None
    if hasattr(shared, "sd_model") and shared.sd_model is not None:
        try:
            print(f"[TensorRT] Zwalnianie VRAM: przenoszenie modelu bazowego do RAM...", flush=True)
            model = shared.sd_model.cpu()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[TensorRT] Uwaga: nie udało się przenieść modelu do CPU: {e}", flush=True)

    try:
        s = time.time()
        ret = engine.build(
            onnx_path,
            use_fp16,
            enable_refit=True,
            enable_preview=True,
            timing_cache=timing_cache,
            input_profile=[profile],
        )
        e = time.time()
        print(f"[TensorRT] Czas kompilacji silnika TensorRT: {(e-s):.2f}s", flush=True)
        return ret
    finally:
        if model is not None:
            try:
                print(f"[TensorRT] Przywracanie modelu do pamięci GPU...", flush=True)
                shared.sd_model = model.cuda()
            except Exception as e:
                print(f"[TensorRT] Uwaga: nie udało się przywrócić modelu do GPU: {e}", flush=True)
