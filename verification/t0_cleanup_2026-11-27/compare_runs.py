"""Compare actual fresh-kernel executions and preserve machine-readable evidence."""
import ast
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import subprocess

import nbformat
import numpy as np
import pandas as pd
from PIL import Image

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
manifest=json.loads((OUT/'manifest.json').read_text())
data_audit=json.loads((OUT/'data_audit.json').read_text())

def read_run(name):
    folder=OUT/'runs'/name
    state=json.loads((folder/'execution.json').read_text())
    assert state['status']=='passed',state
    notebook=nbformat.read(folder/'executed.ipynb',as_version=4)
    nbformat.validate(notebook)
    assert not any(o.output_type=='error' for c in notebook.cells for o in c.get('outputs',[]))
    assert all(c.execution_count is not None for c in notebook.cells if c.cell_type=='code')
    return folder,json.loads((folder/'capture/run.json').read_text()),np.load(folder/'capture/arrays.npz')

def image_equal(a,b):
    with Image.open(a) as aa,Image.open(b) as bb:
        x,y=np.asarray(aa.convert('RGBA')),np.asarray(bb.convert('RGBA'))
    return {'same_shape':x.shape==y.shape,'pixels_equal':bool(np.array_equal(x,y)),
            'sha256_equal':hashlib.sha256(a.read_bytes()).hexdigest()==hashlib.sha256(b.read_bytes()).hexdigest()}

checks=[]
comparisons={}
for model,final in [('lr','lr_cleaned_final'),('svm','svm_cleaned')]:
    af,a,aa=read_run(model+'_baseline')
    bf,b,bb=read_run(final)
    arrays_equal={key:key in bb and np.array_equal(aa[key],bb[key]) for key in aa.files}
    assert set(aa.files)==set(bb.files)
    assert all(arrays_equal.values()),[k for k,v in arrays_equal.items() if not v]
    assert a['raw_X_train']==b['raw_X_train'] and a['raw_X_val']==b['raw_X_val']
    assert a['grid']==b['grid']==[.001,.01,.1,1.,10.,100.]
    assert a['best_C']==b['best_C']
    model_params_equal=a['models']==b['models']
    assert model_params_equal
    metric_columns=[k for k in a['results'][0] if k.startswith(('train_','val_'))]+['f1_gap']
    metric_differences={str(ra['value']):{key:(ra[key],rb[key]) for key in metric_columns if ra[key]!=rb[key]}
                        for ra,rb in zip(a['results'],b['results'])}
    assert not any(metric_differences.values()),metric_differences
    for ra,rb in zip(a['results'],b['results']):
        assert {k:v for k,v in ra.items() if k!='fit_time_s'}=={k:v for k,v in rb.items() if k!='fit_time_s'}
    assert a['attack_type_recall']==b['attack_type_recall']
    core=[f'curve_{model}_t0.png',f'confusion_{model}_t0.png']
    plot_details={}
    for name in core:
        snapshot_equal=a['plots'][name]==b['plots'][name]
        pixels=image_equal(af/'workspace/figures'/name,bf/'workspace/figures'/name)
        assert snapshot_equal and pixels['pixels_equal']
        plot_details[name]={'data_axes_labels_legends_styles_exact':snapshot_equal,**pixels}
    historical=pd.read_csv(OUT/'snapshots/results'/f'{model}_t0.csv')
    executed=pd.DataFrame(a['results'])
    historical_errors={key:float(np.max(np.abs(historical[key].to_numpy()-executed[key].to_numpy()))) for key in metric_columns}
    assert max(historical_errors.values())<=1e-15,historical_errors
    historical_images={}
    for name in core:
        previous=OUT/'snapshots/figures'/name
        if previous.is_file():
            historical_images[name]=image_equal(previous,af/'workspace/figures'/name)
    best=a['results'][a['grid'].index(a['best_C'])]
    independent_max_error=0.
    for record,metrics in zip(b['results'],b['models']):
        for split in ['train','val']:
            for key in ['precision','recall','f1','fpr']:
                independent_max_error=max(independent_max_error,abs(record[f'{split}_{key}']-metrics['independent_scores'][split][key]))
    assert independent_max_error<=1e-15
    comparison={
        'baseline_run':model+'_baseline','cleaned_run':final,
        'all_array_comparisons_exact':True,'array_comparisons':arrays_equal,
        'raw_inputs_exact':True,'preprocessed_features_exact':True,
        'all_model_parameters_scalers_categories_feature_names_and_metrics_exact':model_params_equal,
        'all_reported_training_validation_metrics_exact':True,
        'independent_metric_max_absolute_error':independent_max_error,
        'historical_csv_metric_max_absolute_error':max(historical_errors.values()),
        'historical_png_comparisons':historical_images,
        'plots':plot_details,'best_C':a['best_C'],'best_validation_metrics':{k:best[k] for k in metric_columns if k.startswith('val_')},
        'best_validation_confusion':b['models'][b['grid'].index(b['best_C'])]['independent_scores']['val']['confusion'],
        'predictions_compared':int(sum(aa[f'c{i}_{split}_predictions'].size for i in range(6) for split in ['train','val'])),
        'fit_events_baseline':len(a['fit_events']),'fit_events_cleaned':len(b['fit_events']),
        'every_fit_uses_24000_training_rows':all(event['X']['shape'][0]==24000 for event in a['fit_events']+b['fit_events']),
        'load_split_names':sorted(set(event['name'] for event in a['load_events']+b['load_events'])),
        'removed_optional_plots':sorted(set(a['plots'])-set(b['plots'])),
        'execution_status':'PASS',
    }
    comparisons[model]=comparison
    rows=[
        ('Splits and preprocessed features','24,000 train / 6,000 val; 73 features','Exact values, IDs, order and labels match','PASS'),
        ('Settings and six-C sweep','C=0.001,0.01,0.1,1,10,100; all converged','Identical parameters, grid and convergence','PASS'),
        ('Predictions and numerical model outputs','180,000 train+val predictions over six fits','0 differences; scores, coefficients and scalers exact','PASS'),
        ('Selected C and validation F1',f"C={a['best_C']:g}; F1={best['val_f1']:.12f}",f"C={b['best_C']:g}; F1={best['val_f1']:.12f}",'PASS'),
        ('Independent metric calculation','Binary attack-positive precision/recall/F1; normal-class FPR',f'Matches predictions and labels; max error {independent_max_error:g}','PASS'),
        ('Core validation plots','Validation curve and confusion matrix','Data, axes, labels, legends, styles and PNG pixels exact','PASS'),
    ]
    checks.extend({'notebook':model+'_t0.ipynb','check':r[0],'baseline':r[1],'cleaned':r[2],'status':r[3]} for r in rows)

# Preserve and explain pre-correction evidence rather than hiding parameter drift.
_,lb,la=read_run('lr_baseline')
_,pre,pa=read_run('lr_cleaned')
pre_seed_diffs={key:{'baseline':lb['models'][0]['estimator_params'][key],
                     'pre_fix':pre['models'][0]['estimator_params'].get(key)}
                for key in lb['models'][0]['estimator_params']
                if lb['models'][0]['estimator_params'][key]!=pre['models'][0]['estimator_params'].get(key)}
assert pre_seed_diffs=={'random_state':{'baseline':42,'pre_fix':None}}
assert all(np.array_equal(la[k],pa[k]) for k in la.files)
# Independently exercise only the restored failure guard, with a fake result row.
# This executes no fit calls or notebook training cells.
fixed=nbformat.read(OUT/'inputs/lr_cleaned_final.ipynb',as_version=4)
cell=next(c for c in fixed.cells if c.cell_type=='code' and 'results = pd.DataFrame(rows)' in c.source)
body=ast.parse(cell.source).body
start=next(i for i,node in enumerate(body) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='results' for t in node.targets))
namespace={'pd':pd,'rows':[{'train_f1':.5,'val_f1':.5,'converged':False}]}
try:
    exec(compile(ast.Module(body=body[start:],type_ignores=[]),'<convergence-guard-only>','exec'),namespace)
except RuntimeError:
    guard_checked=True
else:
    raise AssertionError('Restored convergence guard did not reject a failed fit.')

# Git corroboration for LR's computational baseline (markdown may differ).
def computational_ast(path):
    n=nbformat.read(path,as_version=4)
    return ast.dump(ast.parse('\n\n'.join(c.source for c in n.cells if c.cell_type=='code')),include_attributes=False)
git_corroboration=computational_ast(OUT/'inputs/lr_baseline.ipynb')==computational_ast(OUT/'inputs/lr_git_head.ipynb')

changed_originals=[]
for name,digest in manifest['original_file_hashes'].items():
    path=ROOT/name
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        changed_originals.append(name)
assert changed_originals==['notebook/lr_t0.ipynb'],changed_originals
for name,digest in manifest['source_hashes'].items():
    assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest,name
input_final=OUT/'inputs/lr_cleaned_final.ipynb'
assert input_final.read_bytes()==(ROOT/'notebook/lr_t0.ipynb').read_bytes()
assert (OUT/'inputs/svm_cleaned.ipynb').read_bytes()==(ROOT/'notebook/svm_t0.ipynb').read_bytes()

report={
 'overall':'Core successful-run functionality, all six training results and core plots preserved after minimal LR corrections.',
 'comparisons':comparisons,'checks':checks,
 'baseline_provenance':{'lr':'Pre-cleanup backup; computational AST also compared with Git HEAD snapshot.',
                        'lr_git_computational_ast_matches_backup':git_corroboration,
                        'svm':'Preserved pre-cleanup backup with recorded outputs; notebook was not tracked in Git.'},
 'fixes':json.loads((OUT/'fixes.json').read_text()),
 'pre_fix_lr_parameter_differences':pre_seed_diffs,
 'pre_fix_lr_predictions_scores_coefficients_and_features_already_exact':True,
 'restored_lr_nonconvergence_guard_exercised_without_training':guard_checked,
 'data_audit':data_audit,
 'execution_failures':[],
 'execution_environment':manifest['environment'],
 'comparison_policy':manifest['comparison_policy'],
 'scope':{'changed_original_files':changed_originals,
          'original_csvs_figures_data_and_baseline_snapshots_unchanged':True,
          'verification_outputs_saved_separately':True,'test_data_not_available_in_run_workspaces':True,
          'dependencies_installed_or_upgraded':False,'execution_host':'Local macOS; not an HPC login node'},
 'limitations':[
   'Equality on the successful fixed split does not prove unbiased evaluation or deployment generalization.',
   'Full-file EDA informed fixed categories and feature choices before validation isolation.',
   '321 validation rows have exact model-input matches in training: 312 same-label and 9 conflicting-label matches; splits were preserved, not redesigned.',
   'Validation selects hyperparameters, so a final untouched test is still required.',
   'The optional SVM margin histogram and other nonessential diagnostics were intentionally removed during cleanup; only the retained core plots are claimed preserved.',
   'Measured fit times differ with runtime conditions and instrumentation; not treated as model-result differences.',
   'SVM retry branches beyond 5,000 iterations were not needed by actual runs; their budgets and parameters were preserved. The LR failure guard was separately exercised without training.',
 ],
}
(OUT/'report.json').write_text(json.dumps(report,indent=2))
with (OUT/'comparison.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=['notebook','check','baseline','cleaned','status'])
    writer.writeheader();writer.writerows(checks)
print(json.dumps({key:{'best_C':value['best_C'],'best_validation_metrics':value['best_validation_metrics'],
                       'prediction_count':value['predictions_compared'],'plots':value['plots'],
                       'historical_pngs':value['historical_png_comparisons'],
                       'independent_metric_error':value['independent_metric_max_absolute_error']}
                  for key,value in comparisons.items()},indent=2))
print('Changed original files:',changed_originals)
print('LR Git computational baseline corroboration:',git_corroboration)
print('Report:',OUT/'report.json')
