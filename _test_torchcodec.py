import os, sys, ctypes

env_lib = os.path.join(sys.prefix, "Library", "bin")
# Also add base conda mingw path - conda ffmpeg needs MinGW runtime DLLs
conda_root = os.path.dirname(os.path.dirname(sys.prefix))  # D:\miniconda3
mingw_bin = os.path.join(conda_root, "Library", "mingw-w64", "bin")
mingw_usr = os.path.join(conda_root, "Library", "usr", "bin")
conda_lib = os.path.join(conda_root, "Library", "bin")

for p in [env_lib, mingw_bin, mingw_usr, conda_lib]:
    if os.path.exists(p) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(p)
        print(f"Added: {p}")

try:
    ctypes.CDLL(os.path.join(env_lib, "avcodec-60.dll"))
    print("avcodec-60.dll: ok")
except Exception as e:
    print(f"avcodec-60.dll: FAIL")

tc_dll = os.path.join(sys.prefix, "Lib", "site-packages", "torchcodec", "libtorchcodec_core6.dll")
try:
    ctypes.CDLL(tc_dll)
    print("libtorchcodec_core6.dll: ok")
except Exception as e:
    print(f"libtorchcodec_core6.dll: FAIL")

# Final test: import torchcodec
try:
    import torchcodec
    print("import torchcodec: ok")
except Exception as e:
    print(f"import torchcodec: FAIL - {str(e)[:100]}")
