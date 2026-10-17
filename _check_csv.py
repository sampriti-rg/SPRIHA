import csv
path = "training/data/sample_services.csv"
rows = list(csv.DictReader(open(path, encoding="utf-8-sig", errors="replace")))
print(f"Rows loaded : {len(rows)}")
print(f"Columns     : {list(rows[0].keys())}")
print()
hdr = f"{'ID':<3}  {'Name':<26}  {'Sz':>2}  {'Pop':>4}  {'Criticality':<18}  {'Rev':>4}  {'InitCycles':>12}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(
        f"{r['service_id']:<3}  {r['name']:<26}  {r['size']:>2}  "
        f"{float(r['popularity']):>4.2f}  {r['criticality']:<18}  "
        f"{float(r['revenue']):>4.2f}  {float(r['init_cpu_cycles']):>12,.0f}"
    )
