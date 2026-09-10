from __future__ import annotations
import json,threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from idv_agent.data_tools.session_store import SessionStore
from idv_agent.data_tools.workbench import DataWorkbench
from test_mvp_v6_annotations import _raw

def start_server(root):
 from idv_agent.scripts.data_workbench import create_server
 server=create_server(root,'127.0.0.1',0);threading.Thread(target=server.serve_forever,daemon=True).start();return server

def request(server,path,method='GET',payload=None):
 body=None if payload is None else json.dumps(payload).encode();req=Request(f'http://127.0.0.1:{server.server_port}{path}',data=body,method=method,headers={'Content-Type':'application/json'} if body else {})
 try:return urlopen(req,timeout=3)
 except HTTPError as e:return e

def test_workbench_v6_summary_is_the_only_pipeline_summary(tmp_path):
 root=tmp_path/'raw_sessions';_raw(root);w=DataWorkbench(SessionStore(root));s=w.mvp_v6_summary('ep');assert s['schema']=='idv.workbench.v6';assert s['raw_errors']==[];assert set(s['workspaces'])=={'prompt_q','xany_import','nav_review'};assert not hasattr(w,'extract_actions');assert not hasattr(w,'build_vla_chunks_v5')

def test_workbench_readiness_reports_live_cross_session_blockers(tmp_path):
 root=tmp_path/'raw_sessions';_raw(root);w=DataWorkbench(SessionStore(root));assert w.annotation_readiness()['total']==0

def test_v6_prepare_import_and_q_export_are_versioned(tmp_path):
 root=tmp_path/'raw_sessions';session=_raw(root)
 from PIL import Image
 for image in (session/'frames').glob('*.jpg'):
  Image.new('RGB',(1,1)).save(image);image.with_suffix('.json').write_text(json.dumps({'imagePath':image.name,'imageWidth':1,'imageHeight':1,'shapes':[]}))
 w=DataWorkbench(SessionStore(root));assert w.prepare_mvp_v6('ep',output=str(tmp_path/'prompt_q'))['rows']>0
 assert w.import_mvp_v6('ep',workspace=str(tmp_path/'prompt_q'),output=str(tmp_path/'imported'),annotator='user',completion_note='done',split='train',scenario_group='scene')['frames']==50
 assert w.export_q_mvp_v6('ep',workspace=str(tmp_path/'imported'),output=str(tmp_path/'q.jsonl'))['exported_q_rows']==14

def test_v6_http_routes_expose_no_v5_operations(tmp_path):
 root=tmp_path/'raw_sessions';_raw(root);server=start_server(root)
 try:
  assert request(server,'/api/sessions').status==200;assert request(server,'/api/mvp-v6/readiness').status==200;assert json.loads(request(server,'/api/sessions/ep/mvp-v6').read())['schema']=='idv.workbench.v6'
  for path in ('/api/sessions/ep/actions','/api/sessions/ep/intents/init','/api/sessions/ep/chunks/build','/api/camera-control/sets'):assert request(server,path).status in {404,405}
 finally:server.shutdown();server.server_close()

def test_v6_http_prepare_requires_explicit_new_output(tmp_path):
 root=tmp_path/'raw_sessions';_raw(root);server=start_server(root)
 try:
  assert request(server,'/api/sessions/ep/mvp-v6/prepare',method='POST',payload={'output':str(tmp_path/'prepared')}).status==200
  assert request(server,'/api/sessions/ep/mvp-v6/prepare',method='POST',payload={'output':str(tmp_path/'prepared')}).status==409
 finally:server.shutdown();server.server_close()

def test_session_store_still_provides_read_only_frame_access(tmp_path):
 root=tmp_path/'raw_sessions';_raw(root);server=start_server(root)
 try:assert request(server,'/api/sessions/ep/frames/0').status==200
 finally:server.shutdown();server.server_close()
