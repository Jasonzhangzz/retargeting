#!/usr/bin/env python3
"""Package local Three.js, frozen model download and comparison page."""
import argparse,json,shutil,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();web=a.output/'web';web.mkdir(parents=True,exist_ok=True)
 for name in ('index.html','app.js','style.css'):shutil.copy2(ROOT/'web/grasp_compare'/name,web/name)
 vendor=web/'vendor';vendor.mkdir(exist_ok=True)
 for name in ('three.module.min.js','OrbitControls.js'):shutil.copy2(ROOT/'moxian_web/static/vendor'/name,vendor/name)
 with zipfile.ZipFile(web/'saved-left.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
  for f in (a.output/'saved_left').iterdir():
   if f.is_file():z.write(f,'saved_left/'+f.name)
  for f in (a.output/'exports').glob('*'):
   if f.is_file():z.write(f,'exports/'+f.name)
 provenance={'snapshot':json.loads((a.output/'saved_left/manifest.json').read_text()),'appearance':json.loads((a.output/'saved_left/appearance.json').read_text()),'generation':json.loads((a.output/('MANIFEST.json' if (a.output/'MANIFEST.json').exists() else 'GRID_MANIFEST.json')).read_text()),'refinement':json.loads((a.output/'REFINEMENT_MANIFEST.json').read_text()) if (a.output/'REFINEMENT_MANIFEST.json').exists() else None}
 (web/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2));print(web.resolve())
if __name__=='__main__':main()
