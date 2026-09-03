import sys
import re

python = sys.executable

# Provide fallback for standalone execution outside WebUI
try:
    import launch
except ImportError:
    import subprocess

    class MockLaunch:
        @staticmethod
        def is_installed(package):
            try:
                from importlib.metadata import version
                version(package)
                return True
            except Exception:
                return False

        @staticmethod
        def run_pip(args, desc="", live=True):
            print(f"[TensorRT Extension] Running pip {args} ({desc})")
            cmd = [sys.executable, "-m", "pip"] + args.split()
            subprocess.run(cmd, check=True)

        @staticmethod
        def run(cmd, desc=""):
            print(f"[TensorRT Extension] Running command: {cmd} ({desc})")
            subprocess.run(cmd, check=True)

    launch = MockLaunch()


def get_installed_version(package_name):
    try:
        from importlib.metadata import version
        return version(package_name)
    except Exception:
        try:
            import importlib_metadata
            return importlib_metadata.version(package_name)
        except Exception:
            return None


def parse_version(v_str):
    if not v_str:
        return (0, 0, 0)
    parts = [int(p) for p in re.findall(r"\d+", v_str)]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


MIN_TRT_VERSION = (10, 8, 0)  # TensorRT 10.8.0 introduces native Blackwell (sm_120) support


def install():
    # Remove obsolete CUDA 11 cuDNN if left from previous old versions
    if launch.is_installed("nvidia-cudnn-cu11"):
        print("[TensorRT Extension] Removing legacy nvidia-cudnn-cu11...")
        launch.run(
            [python, "-m", "pip", "uninstall", "-y", "nvidia-cudnn-cu11"],
            "removing legacy nvidia-cudnn-cu11",
        )

    # Check TensorRT
    trt_pkg = None
    trt_ver = None
    if launch.is_installed("tensorrt"):
        trt_pkg = "tensorrt"
        trt_ver = get_installed_version("tensorrt")
    elif launch.is_installed("tensorrt-cu12"):
        trt_pkg = "tensorrt-cu12"
        trt_ver = get_installed_version("tensorrt-cu12")

    needs_trt_install = False
    if trt_pkg and trt_ver:
        parsed_ver = parse_version(trt_ver)
        if parsed_ver >= MIN_TRT_VERSION:
            print(
                f"[TensorRT Extension] Found {trt_pkg}=={trt_ver} (>= 10.8.0), fully compatible with Blackwell (sm_120)."
            )
        else:
            print(
                f"[TensorRT Extension] Installed {trt_pkg}=={trt_ver} is too old for Blackwell sm_120 (requires >= 10.8.0). Upgrading..."
            )
            launch.run(
                [python, "-m", "pip", "uninstall", "-y", trt_pkg],
                f"removing outdated {trt_pkg}",
            )
            needs_trt_install = True
    else:
        needs_trt_install = True

    if needs_trt_install:
        print("[TensorRT Extension] Installing TensorRT >= 10.8.0 for CUDA 12 / Blackwell...")
        launch.run_pip(
            "install tensorrt>=10.8.0 --extra-index-url https://pypi.nvidia.com --no-cache-dir",
            "tensorrt",
            live=True,
        )

    # Polygraphy
    if not launch.is_installed("polygraphy"):
        print("[TensorRT Extension] Installing Polygraphy...")
        launch.run_pip(
            "install polygraphy --extra-index-url https://pypi.nvidia.com",
            "polygraphy",
            live=True,
        )

    # ONNX GraphSurgeon (do not pin obsolete protobuf==3.20.2)
    if not launch.is_installed("onnx_graphsurgeon"):
        print("[TensorRT Extension] Installing ONNX-GraphSurgeon...")
        launch.run_pip(
            "install onnx-graphsurgeon --extra-index-url https://pypi.nvidia.com",
            "onnx-graphsurgeon",
            live=True,
        )

    # ONNX
    if not launch.is_installed("onnx"):
        print("[TensorRT Extension] Installing ONNX...")
        launch.run_pip(
            "install onnx",
            "onnx",
            live=True,
        )

    # Optimum
    if not launch.is_installed("optimum"):
        print("[TensorRT Extension] Installing Optimum...")
        launch.run_pip(
            "install optimum",
            "optimum",
            live=True,
        )


if __name__ == "__main__" or "launch" in sys.modules:
    install()

