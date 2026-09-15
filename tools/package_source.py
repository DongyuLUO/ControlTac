"""Package publishable source, excluding Git-ignored experiments and large assets."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

root = Path(__file__).resolve().parents[1]
out = root/'release_assets'
out.mkdir(exist_ok=True)
names = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root).decode().split('\0')
with zipfile.ZipFile(out/'ControlTac-source.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for name in sorted(set(names)-{''}):
        path = root/name
        if path.is_file():
            archive.write(path,'ControlTac/'+name)
with zipfile.ZipFile(out/'ControlTac-source.zip') as archive:
    assert archive.testzip() is None
report = {}
for path in [out/'ControlTac-source.zip',out/'controltac_annotations.zip',root/'checkpoints/force_control.pth',root/'checkpoints/force_pose_control.pth']:
    with path.open('rb') as f:
        sha = hashlib.file_digest(f,'sha256').hexdigest()
    report[path.name] = {'bytes':path.stat().st_size,'sha256':sha}
(out/'manifest.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
