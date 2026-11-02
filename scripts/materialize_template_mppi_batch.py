#!/usr/bin/env python3
"""Keep final artifacts and their exact canonical references in one portable batch."""
import argparse,hashlib,json,os,shutil,sys
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retarget.common.results import write_json


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def relative(p):return str(Path(p).resolve().relative_to(ROOT))


def materialize(source,destination):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if destination.exists():raise ValueError('Use a new destination')
    audit=read(source/'independent_result_audit.json')
    if not audit['passed'] or not audit['complete']:raise ValueError('A complete audited source is required')
    destination.mkdir(parents=True);entries=[];rows=[];references={};copied=[]
    # These small audit files preserve validation history independently of all
    # old experiment directories; runtime/replay never follows old outputs.
    archive=destination/'source_audits';archive.mkdir()
    for name in ('independent_result_audit.json','composition.json','provenance.json','source_sha256_at_start.json'):
        if (source/name).is_file():shutil.copy2(source/name,archive/name)
    for index,original in enumerate(read(source/'manifest.json')['cases']):
        old=(source/original['trajectory']).parent
        new=destination/'cases'/original['name'];new.mkdir(parents=True)
        keep=('scene.xml','input_object.obj','grasp.npz','trajectory.npz','summary.json',
            'projection_force_closure.json','patch_contact_coverage.json','template_contacts.json',
            'initialization.json','size_ik.json','optimization.json','object_frame.json')
        for name in keep:
            if (old/name).is_file():shutil.copy2(old/name,new/name)
        initial=read(old/'initialization.json');old_ref=Path(initial['reference']).resolve()
        if old_ref not in references:
            new_ref=destination/'template_initialization'/old_ref.parent.name/'reference.json'
            new_ref.parent.mkdir(parents=True)
            for name in ('template_contacts.json','candidates.npz','object_1x.obj','scene_1x.xml',
                         'canonical_grasp.json','opposed_pinch_bootstrap.json','object_frame.json'):
                if (old_ref.parent/name).is_file():shutil.copy2(old_ref.parent/name,new_ref.parent/name)
            ref=read(old_ref)
            if ref.get('canonical_grasp_reference'):
                evidence=dict(ref['canonical_grasp_reference']);evidence_file=Path(evidence['source'])
                copied_evidence=new_ref.with_name('canonical_source'+evidence_file.suffix)
                shutil.copy2(evidence_file,copied_evidence)
                evidence.update(source=relative(copied_evidence),sha256=sha(copied_evidence))
                ref['canonical_grasp_reference']=evidence
            write_json(new_ref,ref);references[old_ref]=new_ref
        new_ref=references[old_ref]
        initial.update(reference=relative(new_ref),reference_sha256=sha(new_ref),
            canonical_grasp_reference=read(new_ref).get('canonical_grasp_reference'))
        write_json(new/'initialization.json',initial)
        # MuJoCo resolves mesh files against the XML location, so use relative
        # paths for every asset. This survives relocation to any repository root.
        tree=ET.parse(new/'scene.xml')
        for element in tree.iter():
            if 'file' not in element.attrib:continue
            p=Path(element.attrib['file'])
            if not p.is_absolute():p=(old/p).resolve()
            try:target=new/p.relative_to(old)
            except ValueError:
                p.relative_to(ROOT)  # Refuse non-repository external assets.
                target=p
            if not target.exists():raise FileNotFoundError(target)
            element.attrib['file']=os.path.relpath(target,new)
        tree.write(new/'scene.xml',encoding='unicode')
        e=dict(original)
        for field,file in (('model','scene.xml'),('trajectory','trajectory.npz'),('summary','summary.json')):
            e[field]=str((new/file).relative_to(destination))
        e['patch_coverage']=str((new/'patch_contact_coverage.json').relative_to(destination))
        e.pop('source_batch',None);e.pop('source_case_name',None)
        entries.append(e);rows.append(read(new/'summary.json'))
        copied.append(dict(case=e['name'],original_grasp_sha256=sha(old/'grasp.npz'),
            materialized_grasp_sha256=sha(new/'grasp.npz'),original_projection_sha256=sha(old/'projection_force_closure.json'),
            materialized_projection_sha256=sha(new/'projection_force_closure.json')))
    write_json(destination/'manifest.json',dict(version=1,task='grasp',method='mppi',cases=entries))
    summary=read(source/'summary.json');summary['cases']=rows;write_json(destination/'summary.json',summary)
    ref_report=read(source/'template_initialization/manifest.json')
    ref_report['references']=[relative(references[Path(read((source/e['trajectory']).parent/'initialization.json')['reference']).resolve()])
        for e in read(source/'manifest.json')['cases']]
    write_json(destination/'template_initialization/manifest.json',ref_report)
    for name in ('resources.json','skipped.json','config.yaml','status.json','source_sha256_at_start.json'):
        if (source/name).is_file():shutil.copy2(source/name,destination/name)
    lineage={str(archive/'independent_result_audit.json'):sha(archive/'independent_result_audit.json')}
    write_json(destination/'composition.json',dict(policy='materialized final results and canonical references',source_audit_sha256=lineage))
    write_json(destination/'provenance.json',dict(kind='portable_final_materialization',source_audit_sha256=lineage))
    write_json(destination/'materialization.json',dict(source=str(source),cases=len(entries),references=len(references),
        preserved_final_poses=all(r['original_grasp_sha256']==r['materialized_grasp_sha256'] for r in copied),
        preserved_projection_witnesses=all(r['original_projection_sha256']==r['materialized_projection_sha256'] for r in copied),
        cases_detail=copied,asset_paths='relative to each scene.xml; runtime references relative to repository root'))
    print(destination,len(entries),'cases',len(references),'references')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('destination',type=Path)
    a=p.parse_args();materialize(a.source,a.destination)
