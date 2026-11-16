"""Exercise real HTTP asset delivery, including every page script dependency."""
from html.parser import HTMLParser
from pathlib import Path
import subprocess
import urllib.request
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def web_url():
    interpreter=ROOT/'.venv-moxian/bin/python'
    if not interpreter.exists():pytest.skip('WebUI environment not installed')
    code='''from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from moxian_web.server import handler
server=ThreadingHTTPServer(('127.0.0.1',0),handler(SimpleNamespace()))
print(server.server_address[1],flush=True)
server.serve_forever()
'''
    proc=subprocess.Popen([str(interpreter),'-u','-c',code],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        port=int(proc.stdout.readline().strip());yield f'http://127.0.0.1:{port}'
    finally:
        proc.terminate();proc.wait(timeout=10)


def test_all_html_scripts_styles_and_local_module_imports_are_served(web_url):
    import re
    from urllib.parse import urljoin
    class Assets(HTMLParser):
        def __init__(self):super().__init__();self.paths=[]
        def handle_starttag(self,tag,attrs):
            attrs=dict(attrs)
            if tag=='script' and attrs.get('src'):self.paths.append(attrs['src'])
            if tag=='link' and attrs.get('rel')=='stylesheet':self.paths.append(attrs['href'])
    with urllib.request.urlopen(web_url) as response:html=response.read().decode()
    parser=Assets();parser.feed(html)
    assert any(path.startswith('/geort.js') for path in parser.paths)
    assert any(path.startswith('/cart.js') for path in parser.paths)
    queue=list(parser.paths);visited=set()
    while queue:
        path=queue.pop();url=urljoin(web_url,path)
        if url in visited:continue
        visited.add(url)
        with urllib.request.urlopen(url) as response:
            assert response.status==200
            content=response.read().decode()
            if '.js' in path:assert 'javascript' in response.headers['Content-Type']
        for dep in re.findall(r"(?:from\s*|import\s*)['\"]([^'\"]+)['\"]",content):
            if dep.startswith(('/','./','../')):queue.append(urljoin(url,dep))
    assert len(visited)>=10


def test_recorded_source_identity_validation_runs_in_web_environment_without_scipy():
    code='''from moxian_web.source_ik_initialization import runtime_revision,validate_recorded_source
profile=dict(model_hash='test-model',bone_extrinsics={'schema_revision':'test'})
validate_recorded_source(profile,dict(skeleton_runtime_revision=runtime_revision(profile)))
print('ok')
'''
    result=subprocess.run([str(ROOT/'.venv-moxian/bin/python'),'-c',code],cwd=ROOT,
                          capture_output=True,text=True,check=True)
    assert result.stdout.strip()=='ok'
