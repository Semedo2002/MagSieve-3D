"""Build the native extension src/aegis_core.cpp into the aegis package.

Uses a system C++ compiler if there is one (c++, g++, clang++), else the compiler in the `ziglang` pip package, which
needs no install rights. No CUDA toolkit or MPI headers are needed: the extension loads both at run time.

    pip install pybind11 ziglang
    python tools/build_core.py
"""
import os
import sys
import shutil
import sysconfig
import subprocess

import pybind11

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    src = os.path.join(ROOT, "src", "aegis_core.cpp")
    ext = sysconfig.get_config_var("EXT_SUFFIX") or ".pyd"
    out = os.path.join(ROOT, "aegis", "aegis_core" + (".pyd" if os.name == "nt" else ext))
    inc = [pybind11.get_include(), sysconfig.get_path("include")]
    cc = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)
    if cc:
        cmd = [cc]
    else:
        cmd = [sys.executable, "-m", "ziglang", "c++"]
        if os.name == "nt":
            cmd += ["-target", "x86_64-windows-gnu"]
    cmd += ["-std=c++17", "-O2", "-shared", "-fPIC" if os.name != "nt" else "-DWIN32_LEAN_AND_MEAN"]
    cmd += [f"-I{i}" for i in inc] + [src, "-o", out]
    if os.name == "nt":
        base = sys.base_prefix
        cmd += [f"-L{os.path.join(base, 'libs')}", f"-lpython{sys.version_info.major}{sys.version_info.minor}"]
    else:
        cmd += ["-ldl"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)
    for junk in (out[:-4] + ".pdb", out[:-4] + ".lib"):
        if os.path.exists(junk):
            os.remove(junk)
    print(f"built {out}")


if __name__ == "__main__":
    main()
