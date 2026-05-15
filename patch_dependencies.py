import pathlib
import inspect
import re
import site

import xlstm


def patch_xlstm() -> None:
    base = pathlib.Path(inspect.getfile(xlstm)).parent
    p = base / "xlstm_large" / "model.py"

    print("Patching xLSTM:", p)

    txt = p.read_text()

    if "PATCH: robust unpacking" in txt:
        print("xLSTM already patched")
        return

    pattern = re.compile(
        r"""
        (?P<indent>^[ \t]*)
        h,\s*state\s*=\s*self\.mlstm_backend\s*\(
        (?P<body>[\s\S]*?)
        ^(?P=indent)\)
        """,
        re.MULTILINE | re.VERBOSE,
    )

    m = pattern.search(txt)
    if not m:
        raise RuntimeError("Could not find full mlstm_backend block to patch")

    indent = m.group("indent")
    body = m.group("body")

    replacement = f"""{indent}out = self.mlstm_backend(
{body}
{indent})

{indent}# --- PATCH: robust unpacking for backends that return >2 values ---
{indent}if isinstance(out, tuple):
{indent}    h = out[0]
{indent}    if len(out) == 2:
{indent}        state = out[1]
{indent}    elif len(out) >= 4:
{indent}        state = tuple(out[-3:])
{indent}    else:
{indent}        state = None
{indent}else:
{indent}    h = out
{indent}    state = None
{indent}# --- END PATCH ---
"""

    txt = pattern.sub(replacement, txt, count=1)
    p.write_text(txt)

    print("xLSTM patched successfully")


def verify_xlstm_patch() -> None:
    base = pathlib.Path(inspect.getfile(xlstm)).parent
    p = base / "xlstm_large" / "model.py"

    txt = p.read_text()

    assert "PATCH: robust unpacking" in txt, "xLSTM patch marker not found"
    assert "out = self.mlstm_backend" in txt, "patched backend call not found"
    assert "h, state = self.mlstm_backend" not in txt, "old unpacking still present"

    print("xLSTM patch verified successfully")


def patch_guacamol() -> None:
    for base in site.getsitepackages():
        p = pathlib.Path(base) / "guacamol/utils/chemistry.py"

        if p.exists():
            print("Patching GuacaMol:", p)

            txt = p.read_text()
            txt = txt.replace(
                "from scipy import histogram",
                "from numpy import histogram",
            )
            p.write_text(txt)

            print("GuacaMol patched successfully")
            return

    raise RuntimeError("guacamol/utils/chemistry.py not found")


def verify_guacamol_patch() -> None:
    for base in site.getsitepackages():
        p = pathlib.Path(base) / "guacamol/utils/chemistry.py"

        if p.exists():
            txt = p.read_text()

            assert "from numpy import histogram" in txt, "GuacaMol numpy histogram patch not found"
            assert "from scipy import histogram" not in txt, "old scipy histogram import still present"

            print("GuacaMol patch verified successfully")
            return

    raise RuntimeError("guacamol/utils/chemistry.py not found")


if __name__ == "__main__":
    patch_xlstm()
    verify_xlstm_patch()

    patch_guacamol()
    verify_guacamol_patch()