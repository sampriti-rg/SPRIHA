p = "training/finetune.py"
data = open(p, "rb").read()
# Replace the Windows-1252 em-dash (0x97) with a plain hyphen
data = data.replace(b"\x97", b"-")
open(p, "wb").write(data)
import py_compile
try:
    py_compile.compile(p, doraise=True)
    print("Syntax OK")
except py_compile.PyCompileError as e:
    print("Syntax error:", e)
