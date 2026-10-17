import re

path = 'training/collect_stats.py'
content = open(path, encoding='utf-8-sig', errors='replace').read()

old = '''\ndef load_sample_services_with_names(csv_path: str):
"""
Load services and return (catalogue_dict, name_lookup_dict).

Reads the optional init_cpu_cycles column introduced in the
updated sample_services.csv.  Falls back to the Service dataclass
default (50_000) when the column is absent so older CSV files
remain compatible.
"""
catalogue: Dict[int, Service] = {}
names: Dict[int, str] = {}
with open(csv_path, newline="", encoding="utf-8-sig", errors="replace") as f:
    reader = csv.DictReader(f)
    for row in reader:
        svc_id = int(row["service_id"])
        svc = Service(
            service_id      = svc_id,
            size            = int(row["size"]),
            popularity      = float(row["popularity"]),
            criticality     = row["criticality"].strip(),
            revenue         = float(row["revenue"]),
            init_cpu_cycles = float(row["init_cpu_cycles"])
                              if "init_cpu_cycles" in row and row["init_cpu_cycles"]
                              else 50_000,
        )
        catalogue[svc_id] = svc
        names[svc_id] = row.get("name", f"service_{svc_id}").strip()
return catalogue, names'''

new = '''

def load_sample_services_with_names(csv_path: str):
    """
    Load services and return (catalogue_dict, name_lookup_dict).

    Reads the optional init_cpu_cycles column introduced in the
    updated sample_services.csv.  Falls back to the Service dataclass
    default (50_000) when the column is absent so older CSV files
    remain compatible.
    """
    catalogue: Dict[int, Service] = {}
    names: Dict[int, str] = {}
    with open(csv_path, newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            svc_id = int(row["service_id"])
            svc = Service(
                service_id      = svc_id,
                size            = int(row["size"]),
                popularity      = float(row["popularity"]),
                criticality     = row["criticality"].strip(),
                revenue         = float(row["revenue"]),
                init_cpu_cycles = float(row["init_cpu_cycles"])
                                  if "init_cpu_cycles" in row and row["init_cpu_cycles"]
                                  else 50_000,
            )
            catalogue[svc_id] = svc
            names[svc_id] = row.get("name", f"service_{svc_id}").strip()
    return catalogue, names'''

if old in content:
    content = content.replace(old, new, 1)
    open(path, 'w', encoding='utf-8').write(content)
    print("Fixed OK")
else:
    print("Pattern not found — showing context around line 148:")
    for i, l in enumerate(content.splitlines()[145:170], start=146):
        print(f'{i:3}: {repr(l)}')
