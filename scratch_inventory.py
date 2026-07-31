import csv
import os
from collections import defaultdict

file_path = r'c:\projects\cancer-paper-mill\repo_files.csv'
files = []
with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Avoid counting the .git folder files
        if '.git\\' in row['FullName'] or '.git/' in row['FullName']:
            continue
        # Avoid counting the generated csv
        if 'repo_files.csv' in row['FullName']:
            continue
        try:
            length = int(row['Length'])
            files.append((row['FullName'], length))
        except:
            pass

# Group by type
types = defaultdict(list)
for path, size in files:
    ext = os.path.splitext(path)[1].lower()
    if not ext:
        if 'README' in path:
            ext = 'docs'
        elif '__pycache__' in path:
            ext = 'junk'
        else:
            ext = 'other'
    
    # Custom grouping
    if ext in ['.py', '.sh']:
        group = 'scripts'
    elif ext in ['.ipynb']:
        group = 'notebooks'
    elif ext in ['.csv', '.json', '.txt', '.jsonl']:
        group = 'data'
    elif ext in ['.pt', '.bin', '.ckpt']:
        group = 'models'
    elif ext in ['.log']:
        group = 'logs'
    elif ext in ['.png', '.jpg', '.pdf']:
        group = 'figures'
    elif ext in ['.md']:
        group = 'docs'
    elif ext in ['.pkl', '.npy']:
        group = 'cache'
    else:
        group = 'other'
        
    types[group].append((path, size))

print("Total files:", len(files))
print("Total size (MB):", sum(s for _, s in files) / (1024 * 1024))

for group, group_files in types.items():
    group_size = sum(s for _, s in group_files) / (1024 * 1024)
    print(f"\nGroup: {group} - {len(group_files)} files, {group_size:.2f} MB")
    # sort by size
    group_files.sort(key=lambda x: x[1], reverse=True)
    for path, size in group_files[:20]: # show top 20
        rel_path = os.path.relpath(path, r'c:\projects\cancer-paper-mill')
        print(f"  {size / (1024 * 1024):.2f} MB - {rel_path}")

