"""Exercise slider updates over Viser's websocket and detect mesh rebuilds."""
import argparse,asyncio,json
from urllib.request import urlopen
import msgspec,viser,websockets,zstandard


def decode(data):
    count=int.from_bytes(data[8:16],'little')
    return msgspec.msgpack.decode(zstandard.ZstdDecompressor().decompress(data[16:16+count]))['messages']


async def collect(ws,quiet=.4):
    messages=[]
    while True:
        try:messages.extend(decode(await asyncio.wait_for(ws.recv(),quiet)))
        except asyncio.TimeoutError:return messages


async def check(port):
    for attempt in range(40):
        try:
            with urlopen(f'http://localhost:{port}',timeout=.25):break
        except OSError:
            if attempt==39:raise
            await asyncio.sleep(.25)
    async with websockets.connect(f'ws://localhost:{port}',subprotocols=['viser-v'+viser.__version__],max_size=100_000_000) as ws:
        await ws.send(msgspec.msgpack.encode(dict(type='ViewerCameraMessage',wxyz=[1.,0.,0.,0.],position=[0.,0.,1.],fov=.8,near=.01,far=100.,image_height=800,image_width=1000,look_at=[0.,0.,0.],up_direction=[0.,1.,0.])))
        initial=[]
        while True:
            initial.extend(decode(await asyncio.wait_for(ws.recv(),30)))
            if any(m['type']=='GuiUpdateMessage' and '最终抓取结果' in str(m.get('updates',{})) for m in initial):break
        initial.extend(await collect(ws))
        sliders=[m for m in initial if m['type']=='GuiSliderMessage' and m.get('props',{}).get('label')=='尺寸倍率']
        assert len(sliders)==1,sliders
        uuid=sliders[0]['uuid']
        messages=[]
        for value in (.8,1.3,1.00125):
            await ws.send(msgspec.msgpack.encode(dict(type='GuiUpdateMessage',uuid=uuid,updates=dict(value=value))))
            messages.extend(await collect(ws))
        for i in range(20):
            await ws.send(msgspec.msgpack.encode(dict(type='GuiUpdateMessage',uuid=uuid,updates=dict(value=.7+i*.03))))
        messages.extend(await collect(ws,quiet=1.))
        rebuilt=[m for m in messages if m['type'] in ('MeshMessage','RemoveSceneNodeMessage')]
        scales=[m for m in messages if m['type']=='SceneNodeUpdateMessage' and 'scale' in m['updates']]
        cameras=[m for m in messages if m['type'] in ('SetCameraPositionMessage','SetCameraLookAtMessage','SetCameraUpDirectionMessage')]
        assert not cameras,'Slider changed the camera instead of the display transform'
        errors=[m for m in messages if m['type']=='GuiUpdateMessage' and '加载失败' in str(m['updates'])]
        assert not rebuilt,[(m['type'],m.get('name')) for m in rebuilt]
        assert not errors,errors
        assert len(scales)>=3,'No in-place object resizing detected'
        report=dict(passed=True,initial_meshes=sum(m['type']=='MeshMessage' for m in initial),
            slider_updates_sent=23,mesh_creations_during_drag=0,mesh_removals_during_drag=0,
            in_place_scale_updates=len(scales),scene_transform_updates=sum(m['type'] in ('SetPositionMessage','SetOrientationMessage','SceneNodeUpdateMessage') for m in messages))
        dropdown=next(m for m in initial if m['type']=='GuiDropdownMessage' and m.get('props',{}).get('label')=='抓型')
        family_checks=[]
        for name,values,axis in [('18_Extensior_Type',(1.01,1.111,1.999),'height'),
                                 ('24_Tip_Pinch',(.015,.35,1.2),'radius'),
                                 ('26_Sphere_4_Finger',(.81,1.031,1.49),'uniform')]:
            await ws.send(msgspec.msgpack.encode(dict(type='GuiUpdateMessage',uuid=dropdown['uuid'],updates=dict(value=name))))
            await collect(ws,quiet=1.)
            family_messages=[]
            for value in values:
                await ws.send(msgspec.msgpack.encode(dict(type='GuiUpdateMessage',uuid=uuid,updates=dict(value=value))))
                family_messages.extend(await collect(ws))
            rebuild=[m for m in family_messages if m['type'] in ('MeshMessage','RemoveSceneNodeMessage')]
            changes=[m['updates']['scale'] for m in family_messages if m['type']=='SceneNodeUpdateMessage' and 'scale' in m['updates']]
            assert not rebuild,(name,'rebuilt meshes')
            assert changes,(name,'no object scale updates')
            for x,y,z in changes:
                assert abs(x-y)<1.e-7
                if axis=='height':assert abs(x-1.)<1.e-7
                elif axis=='radius':assert abs(z-1.)<1.e-7
                else:assert abs(x-z)<1.e-7
            assert not [m for m in family_messages if m['type']=='GuiUpdateMessage' and '加载失败' in str(m['updates'])]
            family_checks.append(dict(template=name,scaling_axis=axis,in_place_updates=len(changes),mesh_rebuilds=0))
        report['family_checks']=family_checks
        report['camera_updates_during_drag']=len(cameras)
        print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8089);args=parser.parse_args();asyncio.run(check(args.port))
