"""Same material points, anatomy, world frames and tactile loss on both hands."""
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from moxian_web.tactile_surface import CurvedTaxels
from moxian_web.wuji_surface import WujiSurface,CHAINS,mirror_matrix
from moxian_web.wuji_retarget import TactileRetarget,contact_weights

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def source():
    from moxian_web.hand_model import DEFAULT_MANO
    from moxian_web.mano_runtime import ManoSurface
    from moxian_web.contact import default_mapping
    if not DEFAULT_MANO.exists():pytest.skip('licensed local MANO required')
    m=ManoSurface(DEFAULT_MANO,'cpu');m.set_dimensions(np.zeros(20));m.enable_official_fk()
    m.build_binding({'assignments':default_mapping('left'),'version':'test'})
    return m.topology()


@pytest.fixture(scope='module')
def mapped(source,tmp_path_factory):
    surface=WujiSurface();folder=tmp_path_factory.mktemp('tactile-map');mapping=surface.mapping(source,folder)
    return surface,mapping,folder


def frame(source,stamp=1000):
    j=np.array(source['rest_joints_m']);tips=np.array(source['tip_anchors_m'])
    return {'timestamp_ms':stamp,'tactile_timestamp_ms':stamp,'vertices_m':source['rest_vertices_m'],
            'joints_m':np.vstack((j[0],*[np.vstack((j[list(c)],tips[i]))for i,c in enumerate(CHAINS)])).tolist(),
            'geometry_version':source['geometry_version'],'binding_version':source['binding_version'],
            'source_contact':{'ready':True,'cell_contact':[False]*460,'normalized':[0.]*460}}


def test_python_pn_taxels_match_actual_browser_code(source,tmp_path,record_property):
    vendor=tmp_path/'three.mjs';vendor.write_bytes((ROOT/'moxian_web/static/vendor/three.module.min.js').read_bytes())
    module=tmp_path/'curved.mjs';module.write_text((ROOT/'moxian_web/static/curved-surface.js').read_text().replace("'three'",repr(vendor.as_uri())))
    script="""import {CurvedSurface} from MODULE;let input='';for await(const x of process.stdin)input+=x;const d=JSON.parse(input),s=new CurvedSurface(d.topology),v=s.update(d.vertices);const p=s.topology.sid_face_ids.map((f,i)=>{const ids=s.topology.faces[f],bc=s.topology.sid_barycentric[i];return [0,1,2].map(k=>ids.reduce((a,j,t)=>a+bc[t]*v[3*j+k],0));});process.stdout.write(JSON.stringify(p));""".replace('MODULE',repr(module.as_uri()))
    vertices=np.array(source['rest_vertices_m']);rng=np.random.default_rng(18)
    evaluator=CurvedTaxels(source)
    maximum=0.
    for displacement in (np.zeros_like(vertices),rng.normal(0,.003,vertices.shape),rng.normal(0,.012,vertices.shape)):
        v=vertices+displacement
        out=subprocess.run(['node','--input-type=module','-e',script],input=json.dumps({'topology':source,'vertices':v.tolist()}),text=True,capture_output=True,check=True)
        actual=json.loads(out.stdout);expected,normals=evaluator.evaluate(v)
        maximum=max(maximum,float(np.max(np.linalg.norm(expected-actual,axis=1))))
        assert maximum<.00002
        np.testing.assert_allclose(np.linalg.norm(normals,axis=1),1,atol=1e-7)
    record_property('max_browser_pn_error_mm',maximum*1000)


def test_all_460_are_distinct_fixed_on_correct_safe_native_links(source,mapped):
    surface,mapping,folder=mapped
    assert mapping['ready'] and not mapping['failed_sids']
    records=mapping['records'];assert len(records)==460 and [r['sid']for r in records]==list(range(460))
    points=np.array([r['local_position_m']for r in records]);owners=np.array([r['owner']for r in records])
    for sid,r in enumerate(records):
        p=source['sid_patch_indices'][sid];assert r['patch']==p
        if p>=14:assert r['owner']==0
        elif p<2:assert r['owner'] in ([4] if p==0 else [2,3])
        else:assert r['owner']==6+4*((p-2)//3)+(4-p)%3
        assert r['normal_error_deg']<=60.0001 and r['outer_visible'] and r['footprint_m']==.0015
        bc=np.array(r['barycentric']);assert min(bc)>=0 and abs(bc.sum()-1)<1e-8
        np.testing.assert_allclose(bc@surface.local[surface.faces[r['face_id']]],r['local_position_m'],atol=1e-9)
    assert len(np.unique(np.c_[owners,np.round(points,8)],axis=0))==460
    assert surface.mapping(source,folder)==mapping
    M=mirror_matrix(source);np.testing.assert_allclose(M@M,np.eye(3),atol=1e-12);assert np.linalg.det(M)==pytest.approx(-1)
    changed=copy.deepcopy(source);changed['geometry_version']='new-sized-model';assert surface.mapping(changed,folder)['version']!=mapping['version']


def test_binary_activation_relative_response_and_unknown_are_independent():
    patches=np.arange(460)%20;contact={'ready':True,'cell_contact':[False]*460,'normalized':[0.]*460}
    contact['cell_contact'][0]=contact['cell_contact'][20]=True;contact['normalized'][0]=.25;contact['normalized'][20]=1
    w=contact_weights(contact,patches,1000,1000)
    assert w['mask'].sum()==2 and w['active'].sum()==1
    assert w['center'][0,20]==pytest.approx(.8) and w['cloud'][0]==pytest.approx(.125)
    assert w['cloud'][20]==pytest.approx(.5)
    for kwargs in ({'enabled':False},{'tactile_timestamp_ms':1026}):
        args={'pose_timestamp_ms':1000,'tactile_timestamp_ms':1000,**kwargs};u=contact_weights(contact,patches,**args)
        assert not u['ready'] and not u['mask'].any() and not u['cloud'].any()
    contact['normalized']=[0.]*460;u=contact_weights(contact,patches,1000,1000)
    assert u['center'][0,0]==pytest.approx(.5) and u['center'][0,20]==pytest.approx(.5)


def test_card_replaces_lm_and_unknown_detector_remains_inactive():
    from moxian_web.card_math import CardSettings
    assert not hasattr(TactileRetarget,'lm')
    settings=CardSettings();assert settings.population==1024 and settings.iterations==16
    contact={'ready':True,'cell_contact':[True]*460,'normalized':[.5]*460,'timestamp_ms':900}
    assert not contact_weights(contact,np.arange(460)%20,1000,1000)['ready']
