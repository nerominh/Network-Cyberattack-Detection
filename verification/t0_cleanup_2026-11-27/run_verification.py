"""Execute preserved input copies in fresh, isolated kernels; keep originals intact."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import nbformat
from nbclient import NotebookClient

OUT=Path(__file__).resolve().parent

def run(name):
    folder=OUT/'runs'/name
    if folder.exists():raise RuntimeError(f'Run already exists: {folder}')
    work=folder/'workspace'
    (work/'notebook').mkdir(parents=True)
    shutil.copytree(OUT/'snapshots/src',work/'src')
    shutil.copytree(OUT/'snapshots/data',work/'data')
    shutil.copytree(OUT/'snapshots/results',work/'results')
    (work/'figures').mkdir()
    shutil.copyfile(OUT/'verification_hooks.py',work/'verification_hooks.py')
    input_path=OUT/'inputs'/f'{name}.ipynb'
    notebook=nbformat.read(input_path,as_version=4)
    source_cells=[cell.source for cell in notebook.cells]
    setup=nbformat.v4.new_code_cell(
        'import sys\nfrom pathlib import Path\n'
        'sys.path.insert(0, str(Path.cwd().parent))\n'
        'import verification_hooks as _verification_hooks\n'
        '_verification_hooks.install_hooks()',metadata={'tags':['verification-only']})
    capture=nbformat.v4.new_code_cell('_verification_hooks.capture(globals())',metadata={'tags':['verification-only']})
    notebook.cells.insert(0,setup)
    notebook.cells.append(capture)
    executed_path=folder/'executed.ipynb'
    state={'name':name,'status':'running','source_cells_preserved':True}
    with tempfile.TemporaryDirectory(prefix='t0-audit-kernel-') as temp:
        share=Path(temp)/'share/jupyter'
        kernel=share/'kernels/t0-verification'
        kernel.mkdir(parents=True)
        (kernel/'kernel.json').write_text(json.dumps({
            'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
            'display_name':'T0 cleanup verification','language':'python'}))
        os.environ['JUPYTER_PATH']=str(share)
        os.environ['MPLCONFIGDIR']='/private/tmp/network-eda-mpl'
        os.environ['MPLBACKEND']='module://matplotlib_inline.backend_inline'
        os.environ['PYTHONDONTWRITEBYTECODE']='1'
        for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
            os.environ[key]='1'
        def checkpoint(cell,cell_index,**kwargs):
            nbformat.write(notebook,executed_path)
            for output in cell.get('outputs',[]):
                if output.output_type=='stream':print(f'[{name}] {output.text}',end='',flush=True)
        try:
            NotebookClient(notebook,timeout=300,kernel_name='t0-verification',
                           resources={'metadata':{'path':str(work/'notebook')}},
                           on_cell_executed=checkpoint).execute()
            assert [cell.source for cell in notebook.cells[1:-1]]==source_cells
            nbformat.validate(notebook)
            state['status']='passed'
        except Exception as exc:
            state.update(status='failed',error=repr(exc))
            raise
        finally:
            nbformat.write(notebook,executed_path)
            (folder/'execution.json').write_text(json.dumps(state,indent=2))
    print('COMPLETED',name,flush=True)

if __name__=='__main__':
    for name in sys.argv[1:] or ['lr_baseline','lr_cleaned','svm_baseline','svm_cleaned']:
        print('STARTING',name,flush=True)
        run(name)
