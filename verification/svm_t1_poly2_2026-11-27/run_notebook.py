import json,os,sys,tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient

out=Path(__file__).resolve().parent;root=out.parents[1]
path=root/'notebook/svm_t1_poly2.ipynb'
nb=nbformat.read(path,as_version=4)
original_sources=[c.source for c in nb.cells]
setup=nbformat.v4.new_code_cell(
    'import sys\nsys.path.insert(0, '+repr(str(out))+')\n'
    'sys.path.insert(0, '+repr(str(root))+')\n'
    'import svm_t1_audit as _svm_t1_audit\n_svm_t1_audit.install()',
    metadata={'tags':['verification-only']})
nb.cells.insert(0,setup)
nb.cells.append(nbformat.v4.new_code_cell('_svm_t1_audit.capture(globals())',metadata={'tags':['verification-only']}))
class ProgressClient(NotebookClient):
    def process_message(self,msg,cell,cell_index):
        if msg['msg_type']=='stream':print(msg['content']['text'],end='',flush=True)
        return super().process_message(msg,cell,cell_index)

with tempfile.TemporaryDirectory(prefix='svm-t1-kernel-') as temporary:
    share=Path(temporary)/'share/jupyter'
    kernel=share/'kernels/svm-t1-validation';kernel.mkdir(parents=True)
    (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'SVM T1 validation','language':'python'}))
    os.environ['JUPYTER_PATH']=str(share)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['MPLCONFIGDIR']='/private/tmp/network-eda-mpl'
    os.environ['MPLBACKEND']='module://matplotlib_inline.backend_inline'
    def checkpoint(cell,cell_index,**kwargs):nbformat.write(nb,out/'executed_with_audit.ipynb')
    status={'status':'running'}
    try:
        ProgressClient(nb,timeout=None,kernel_name='svm-t1-validation',resources={'metadata':{'path':str(path.parent)}},on_cell_executed=checkpoint).execute()
        assert [c.source for c in nb.cells[1:-1]]==original_sources
        nbformat.validate(nb)
        nbformat.write(nb,out/'executed_with_audit.ipynb')
        nb.cells=nb.cells[1:-1]
        for i,cell in enumerate([c for c in nb.cells if c.cell_type=='code'],start=1):
            cell.execution_count=i
            for output in cell.get('outputs',[]):
                if output.output_type=='execute_result':output.execution_count=i
        nbformat.validate(nb)
        nbformat.write(nb,path)
        status={'status':'passed','all_user_cells_executed':True,'audit_passed':True}
    except Exception as error:
        status={'status':'failed','error':repr(error)}
        raise
    finally:
        (out/'execution_status.json').write_text(json.dumps(status,indent=2))
print('Saved executed notebook:',path)
