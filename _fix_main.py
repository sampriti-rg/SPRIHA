p = "main.py"
lines = open(p, encoding="utf-8-sig", errors="replace").readlines()
# Find def _run_train and fix indentation of its body
start = next(i for i, l in enumerate(lines) if l.strip() == "def _run_train(args):")
# Collect body lines (until next def or blank-then-def)
end = start + 1
while end < len(lines):
    stripped = lines[end].strip()
    if stripped.startswith("def ") and not lines[end].startswith(" "):
        break
    end += 1

body = [
    "def _run_train(args):\n",
    "    from training.train import train\n",
    "    kwargs = dict(seed=args.seed, save_plots=not args.no_plots,\n",
    "                  print_every=args.print_every, traffic_model=args.traffic,\n",
    "                  num_episodes=args.episodes)\n",
    "    if args.checkpoint:\n",
    "        kwargs[\"checkpoint_path\"] = args.checkpoint\n",
    "    if getattr(args, \"services\", None):\n",
    "        kwargs[\"services_csv\"] = args.services\n",
    "    train(**kwargs)\n",
    "\n",
]
new_lines = lines[:start] + body + lines[end:]
open(p, "w", encoding="utf-8").write("".join(new_lines))
print("Fixed lines", start+1, "to", end)
# Quick syntax check
import py_compile
try:
    py_compile.compile(p, doraise=True)
    print("Syntax OK")
except py_compile.PyCompileError as e:
    print("Syntax error:", e)
