#
# Copyright 2022 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import copy
import logging
from logging import warning, info, error
import torch
from torch.cuda import nvtx
from collections import OrderedDict
import numpy as np
from polygraphy.backend.common import bytes_from_path
from polygraphy import util
from polygraphy.backend.trt import ModifyNetworkOutputs, Profile
from polygraphy.backend.trt import (
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.logger import G_LOGGER
import tensorrt as trt
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get("total", 0)
            self.n = 0

        def update(self, n=1):
            self.n += n

        def refresh(self):
            pass

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
G_LOGGER.module_severity = G_LOGGER.ERROR

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {
    value: key for (key, value) in numpy_to_torch_dtype_dict.items()
}

class TQDMProgressMonitor(trt.IProgressMonitor):
    def __init__(self):
        trt.IProgressMonitor.__init__(self)
        self._active_phases = {}
        self._step_result = True
        self.max_indent = 5

    def phase_start(self, phase_name, parent_phase, num_steps):
        leave = False
        try:
            if parent_phase is not None:
                nbIndents = (
                    self._active_phases.get(parent_phase, {}).get(
                        "nbIndents", self.max_indent
                    )
                    + 1
                )
                if nbIndents >= self.max_indent:
                    return
            else:
                nbIndents = 0
                leave = True
            self._active_phases[phase_name] = {
                "tq": tqdm(
                    total=num_steps, desc=phase_name, leave=leave, position=nbIndents
                ),
                "nbIndents": nbIndents,
                "parent_phase": parent_phase,
            }
        except KeyboardInterrupt:
            # The phase_start callback cannot directly cancel the build, so request the cancellation from within step_complete.
            self._step_result = False

    def phase_finish(self, phase_name):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    self._active_phases[phase_name]["tq"].total
                    - self._active_phases[phase_name]["tq"].n
                )

                parent_phase = self._active_phases[phase_name].get("parent_phase", None)
                while parent_phase is not None:
                    self._active_phases[parent_phase]["tq"].refresh()
                    parent_phase = self._active_phases[parent_phase].get(
                        "parent_phase", None
                    )
                if (
                    self._active_phases[phase_name]["parent_phase"]
                    in self._active_phases.keys()
                ):
                    self._active_phases[
                        self._active_phases[phase_name]["parent_phase"]
                    ]["tq"].refresh()
                del self._active_phases[phase_name]
            pass
        except KeyboardInterrupt:
            self._step_result = False

    def step_complete(self, phase_name, step):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    step - self._active_phases[phase_name]["tq"].n
                )
            return self._step_result
        except KeyboardInterrupt:
            # There is no need to propagate this exception to TensorRT. We can simply cancel the build.
            return False


class Engine:
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None  # cuda graph

    def __del__(self):
        for attr in ("context", "engine", "buffers", "tensors"):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass

    def reset(self, engine_path=None):
        for attr in ("context", "engine", "buffers", "tensors"):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass
        self.engine_path = engine_path

        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.inputs = {}
        self.outputs = {}

    def refit_from_dict(self, refit_weights, is_fp16):
        # Initialize refitter
        refitter = trt.Refitter(self.engine, TRT_LOGGER)

        refitted_weights = set()
        # In TRT 10: get_all_weights() returns list of weight names
        trt_weight_names = refitter.get_all_weights()
        for trt_weight_name in trt_weight_names:
            if trt_weight_name not in refit_weights:
                continue

            # get weight from state dict
            trt_datatype = trt.DataType.FLOAT
            if is_fp16:
                refit_weights[trt_weight_name] = refit_weights[trt_weight_name].half()
                trt_datatype = trt.DataType.HALF

            refit_weights[trt_weight_name] = refit_weights[trt_weight_name].contiguous()
            trt_wt_tensor = trt.Weights(
                trt_datatype,
                refit_weights[trt_weight_name].data_ptr(),
                torch.numel(refit_weights[trt_weight_name]),
            )
            trt_wt_location = (
                trt.TensorLocation.DEVICE
                if refit_weights[trt_weight_name].is_cuda
                else trt.TensorLocation.HOST
            )

            # In TRT 10, set_named_weights supports (name, weights, location) or (name, weights)
            try:
                refitter.set_named_weights(trt_weight_name, trt_wt_tensor, trt_wt_location)
            except TypeError:
                refitter.set_named_weights(trt_weight_name, trt_wt_tensor)

            refitted_weights.add(trt_weight_name)

        assert set(refitted_weights) == set(refit_weights.keys())
        if not refitter.refit_cuda_engine():
            raise RuntimeError("Error: failed to refit new weights into TensorRT engine.")

        print(f"[I] Total refitted weights {len(refitted_weights)}.")

    def build(
        self,
        onnx_path,
        fp16,
        input_profile=None,
        enable_refit=False,
        enable_preview=False,
        enable_all_tactics=False,
        timing_cache=None,
        update_output_names=None,
    ):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = [Profile()]
        if input_profile:
            p = [Profile() for i in range(len(input_profile))]
            for _p, i_profile in zip(p, input_profile):
                for name, dims in i_profile.items():
                    assert len(dims) == 3
                    _p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}
        if not enable_all_tactics:
            config_kwargs["tactic_sources"] = []

        network_flags = []
        if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
            network_flags.append(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        elif hasattr(trt.OnnxParserFlag, "NATIVE_INSTANCENORM"):
            network_flags.append(trt.OnnxParserFlag.NATIVE_INSTANCENORM)

        network = network_from_onnx_path(
            onnx_path, flags=network_flags
        )
        if update_output_names:
            print(f"Updating network outputs to {update_output_names}")
            network = ModifyNetworkOutputs(network, update_output_names)

        builder = network[0]
        config = builder.create_builder_config()
        config.progress_monitor = TQDMProgressMonitor()

        # In TRT 10 strongly typed mode, precision is determined by the ONNX graph; set FP16 only if not strongly typed
        is_strongly_typed = getattr(network[1], "has_strongly_typed_layers", False) if hasattr(network[1], "has_strongly_typed_layers") else False
        if fp16 and hasattr(trt.BuilderFlag, "FP16") and not is_strongly_typed:
            try:
                config.set_flag(trt.BuilderFlag.FP16)
            except Exception:
                pass

        if enable_refit and hasattr(trt.BuilderFlag, "REFIT"):
            config.set_flag(trt.BuilderFlag.REFIT)

        cache = None
        if timing_cache:
            timing_cache_dir = os.path.dirname(os.path.abspath(timing_cache))
            if timing_cache_dir and not os.path.exists(timing_cache_dir):
                os.makedirs(timing_cache_dir, exist_ok=True)

            try:
                if os.path.exists(timing_cache) and os.path.getsize(timing_cache) > 0:
                    with util.LockFile(timing_cache):
                        timing_cache_data = util.load_file(
                            timing_cache, description="tactic timing cache"
                        )
                        cache = config.create_timing_cache(timing_cache_data)
                else:
                    cache = config.create_timing_cache(b"")
            except Exception as e:
                warning(
                    f"Timing cache error ({e}), initializing fresh empty timing cache."
                )
                try:
                    cache = config.create_timing_cache(b"")
                except Exception:
                    cache = None

            if cache is not None:
                config.set_timing_cache(cache, ignore_mismatch=True)

        profiles = copy.deepcopy(p)
        for profile in profiles:
            calib_profile = profile.fill_defaults(network[1]).to_trt(
                builder, network[1]
            )
            config.add_optimization_profile(calib_profile)

        try:
            print(f"[TensorRT] Kompilacja sieci i przeszukiwanie taktyk jądra (engine_from_network)...", flush=True)
            print(f"[TensorRT] Może to zająć od 2 do 8 minut w zależności od rozdzielczości i modelu. Proszę czekać...", flush=True)
            engine = engine_from_network(
                network,
                config,
                save_timing_cache=timing_cache,
            )
        except Exception as e:
            print(f"[TensorRT BŁĄD] Kompilacja silnika TensorRT nie powiodła się: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return 1
        try:
            print(f"[TensorRT] Zapisywanie skompilowanego silnika do: {self.engine_path}", flush=True)
            save_engine(engine, path=self.engine_path)
            print(f"[TensorRT] Silnik TensorRT został pomyślnie zapisany na dysku!", flush=True)
        except Exception as e:
            print(f"[TensorRT BŁĄD] Zapis pliku silnika nie powiódł się: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return 1
        return 0

    def load(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=False):
        if reuse_device_memory and hasattr(self.engine, "create_execution_context_without_device_memory"):
            try:
                self.context = self.engine.create_execution_context_without_device_memory()
                return
            except Exception:
                pass
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda", additional_shapes=None):
        nvtx.range_push("allocate_buffers")
        # Pass 1: Set shapes for all input tensors first, so dynamic output shapes are resolved by TensorRT
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if shape_dict and name in shape_dict:
                    shape = tuple(shape_dict[name].shape)
                elif additional_shapes and name in additional_shapes:
                    shape = tuple(additional_shapes[name])
                else:
                    shape = tuple(self.context.get_tensor_shape(name))
                self.context.set_input_shape(name, shape)

        # Pass 2: Allocate or reuse buffers for all I/O tensors
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            is_input = (self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
            if is_input and shape_dict and name in shape_dict:
                shape = tuple(shape_dict[name].shape)
            elif additional_shapes and name in additional_shapes:
                shape = tuple(additional_shapes[name])
            else:
                shape = tuple(self.context.get_tensor_shape(name))

            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype = numpy_to_torch_dtype_dict.get(dtype, torch.float32)

            target_device = torch.device(device)
            # Reuse buffer if already allocated with matching shape, dtype, and device
            if (
                name in self.tensors
                and self.tensors[name].shape == shape
                and self.tensors[name].dtype == torch_dtype
                and self.tensors[name].device.type == target_device.type
            ):
                continue

            self.tensors[name] = torch.zeros(shape, dtype=torch_dtype, device=device)
        nvtx.range_pop()

    def infer(self, feed_dict, stream, use_cuda_graph=False):
        nvtx.range_push("set_tensors")
        for name, buf in feed_dict.items():
            if name in self.tensors:
                self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        nvtx.range_pop()

        nvtx.range_push("execute")
        noerror = self.context.execute_async_v3(stream)
        if not noerror:
            raise ValueError("ERROR: TensorRT inference failed.")
        nvtx.range_pop()
        return self.tensors

    def __str__(self):
        try:
            out = ""
            for opt_profile in range(self.engine.num_optimization_profiles):
                out += f"Profile {opt_profile}:\n"
                if hasattr(self.engine, "num_io_tensors"):
                    for binding in range(self.engine.num_io_tensors):
                        name = self.engine.get_tensor_name(binding)
                        shape = "unknown"
                        try:
                            is_input = (
                                hasattr(self.engine, "get_tensor_mode")
                                and self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                            )
                            if is_input and hasattr(self.engine, "get_tensor_profile_shape"):
                                shape = self.engine.get_tensor_profile_shape(name, opt_profile)
                            elif hasattr(self.engine, "get_tensor_shape"):
                                shape = self.engine.get_tensor_shape(name)
                            elif hasattr(self.engine, "get_profile_shape"):
                                shape = self.engine.get_profile_shape(opt_profile, name)
                        except Exception:
                            try:
                                shape = self.engine.get_tensor_shape(name)
                            except Exception:
                                pass
                        out += f"\t{name} = {shape}\n"
                elif hasattr(self.engine, "num_bindings"):
                    for binding_idx in range(self.engine.num_bindings):
                        name = self.engine.get_binding_name(binding_idx)
                        try:
                            shape = self.engine.get_profile_shape(opt_profile, name)
                        except Exception:
                            shape = "unknown"
                        out += f"\t{name} = {shape}\n"
            return out
        except Exception:
            return f"TensorRT Engine ({self.engine_path})"
