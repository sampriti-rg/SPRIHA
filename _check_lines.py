lines = open('training/collect_stats.py', encoding='utf-8-sig', errors='replace').readlines()
for i, l in enumerate(lines[143:165], start=144):
    print(f'{i:3}: {repr(l)}')
