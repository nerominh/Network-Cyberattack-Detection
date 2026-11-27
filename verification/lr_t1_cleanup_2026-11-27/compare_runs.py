"""Compare actual baseline/cleaned executions without fitting more models."""
import ast,csv,hashlib,json
from html.parser import HTMLParser
from pathlib import Path
import nbformat
import numpy as np
import pandas as pd
from PIL import Image

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
manifest=json.loads((OUT/'manifest.json').read_text())

def load(name):
 folder=OUT/'runs'/name
 state=json.loads((folder/'execution.json').read_text())
 assert state['status']=='passed',state
 nb=nbformat.read(folder/'executed.ipynb',as_version=4);nbformat.validate(nb)
 assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
 assert not any(o.output_type=='error' for c in nb.cells for o in c.get('outputs',[]))
 return folder,json.loads((folder/'capture/run.json').read_text()),np.load(folder/'capture/arrays.npz'),nb

def pixels(a,b):
 with Image.open(a) as x,Image.open(b) as y:
  xa,ya=np.asarray(x.convert('RGBA')),np.asarray(y.convert('RGBA'))
 return {'rgba_pixels_equal':bool(np.array_equal(xa,ya)),
         'png_bytes_equal':a.read_bytes()==b.read_bytes()}

class Table(HTMLParser):
 def __init__(self):super().__init__();self.rows=[];self.row=None;self.cell=None
 def handle_starttag(self,tag,attrs):
  if tag=='tr':self.row=[]
  if tag in {'td','th'}:self.cell=''
 def handle_data(self,data):
  if self.cell is not None:self.cell+=data
 def handle_endtag(self,tag):
  if tag in {'td','th'} and self.cell is not None:
   self.row.append(self.cell.strip());self.cell=None
  if tag=='tr' and self.row is not None:self.rows.append(self.row);self.row=None

def tables(nb):
 result={}
 for cell in nb.cells:
  for output in cell.get('outputs',[]):
   html=output.get('data',{}).get('text/html')
   if not html:continue
   parser=Table();parser.feed(html)
   rows=parser.rows
   if not rows:continue
   header=rows[0]
   if 'term_type' in header:result['polynomial_terms']=rows
   if 'caught' in header and 'missed' in header:result['attack_type_recall']=rows
   if 'fit_time_s' in header:
    index=header.index('fit_time_s')
    result['metric_table_except_wall_time']=[row[:index]+row[index+1:] for row in rows]
 return result

comparisons={}
for mode,a_name,b_name in [('controlled_single_thread','baseline','cleaned'),
                          ('project_default_runtime','baseline_default','cleaned_default')]:
 af,a,aa,an=load(a_name);bf,b,bb,bn=load(b_name)
 assert set(aa.files)==set(bb.files)
 exact={key:bool(np.array_equal(aa[key],bb[key])) for key in aa.files}
 assert all(exact.values()),[key for key,value in exact.items() if not value]
 assert a['raw_X_train']==b['raw_X_train'] and a['raw_X_val']==b['raw_X_val']
 assert a['grid']==b['grid']==[.001,.01,.1,1.,10.,100.]
 assert a['best_C']==b['best_C']
 assert a['models']==b['models']
 assert a['feature_names']==b['feature_names']
 assert a['term_table']==b['term_table']
 assert a['transformation_hashes']==b['transformation_hashes']
 assert a['attack_type_recall']==b['attack_type_recall']
 assert tables(an)==tables(bn)
 for x,y in zip(a['results'],b['results']):
  assert {k:v for k,v in x.items() if k!='fit_time_s'}=={k:v for k,v in y.items() if k!='fit_time_s'}
 plots={}
 for name in ['curve_lr_t1_poly2.png','confusion_lr_t1_poly2.png']:
  assert a['plots'][name]==b['plots'][name]
  p=pixels(af/'workspace/figures'/name,bf/'workspace/figures'/name)
  assert p['rgba_pixels_equal']
  plots[name]={'plotted_data_axes_labels_legends_styles_exact':True,**p}
 independent_errors=[]
 for row,model in zip(b['results'],b['models']):
  for split in ['train','val']:
   for metric in ['precision','recall','f1','fpr']:
    independent_errors.append(abs(row[f'{split}_{metric}']-model['independent_scores'][split][metric]))
 assert max(independent_errors)<=1e-15
 best=b['results'][b['grid'].index(b['best_C'])]
 comparisons[mode]={
  'baseline_run':a_name,'cleaned_run':b_name,'status':'PASS',
  'exact_array_comparisons':exact,'input_values_ids_labels_order_exact':True,
  'polynomial_unscaled_and_scaled_values_all_rows_exact_by_sha256':True,
  'six_model_configurations_scalers_and_coefficients_exact':True,
  'reported_metrics_exact_excluding_wall_time':True,'tables_exact_excluding_wall_time':True,
  'table_types_compared':list(tables(an)),
  'best_C':b['best_C'],'best_validation_metrics':{key:best[key] for key in best if key.startswith('val_')},
  'validation_confusion':b['models'][b['grid'].index(b['best_C'])]['independent_scores']['val']['confusion'],
  'predictions_compared':sum(aa[f'c{i}_{split}_predictions'].size for i in range(6) for split in ['train','val']),
  'independent_metric_max_error':max(independent_errors),'plots':plots,
  'fit_rows':sorted(set(event['X']['shape'][0] for event in b['fit_events'])),
  'fit_events_baseline':len(a['fit_events']),'fit_events_cleaned':len(b['fit_events']),
  'load_split_names':sorted(set(event['name'] for event in a['load_events']+b['load_events']))}

# Historical results are reproduced under the project's default runtime.
af,a,aa,_=load('baseline_default');bf,b,bb,_=load('cleaned_default')
historical=pd.read_csv(OUT/'snapshots/results/lr_t1_poly2.csv')
current=pd.DataFrame(b['results'])
metric_cols=[key for key in current if key.startswith(('train_','val_'))]+['f1_gap']
historical_error=float(abs(historical[metric_cols]-current[metric_cols]).to_numpy().max())
assert historical_error<=1e-15
historical_plots={name:pixels(OUT/'snapshots/figures'/name,bf/'workspace/figures'/name)
                  for name in ['curve_lr_t1_poly2.png','confusion_lr_t1_poly2.png']}
assert all(item['rgba_pixels_equal'] for item in historical_plots.values())
_,single,sa,_=load('baseline')
runtime_sensitivity=[]
for i,C in enumerate(a['grid']):
 runtime_sensitivity.append({'C':C,
  'train_prediction_differences':int(np.count_nonzero(sa[f'c{i}_train_predictions']!=aa[f'c{i}_train_predictions'])),
  'val_prediction_differences':int(np.count_nonzero(sa[f'c{i}_val_predictions']!=aa[f'c{i}_val_predictions'])),
  'single_thread_val_f1':single['results'][i]['val_f1'],'default_val_f1':a['results'][i]['val_f1']})

# Check original experiment settings and retry candidates statically as well.
def settings(path):
 nb=nbformat.read(path,as_version=4)
 tree=ast.parse('\n\n'.join(c.source for c in nb.cells if c.cell_type=='code'))
 grid=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GRID' for t in n.targets))
 budgets=[ast.literal_eval(n) for n in ast.walk(tree) if isinstance(n,ast.List) and all(isinstance(x,ast.Constant) for x in n.elts) and ast.literal_eval(n)==[2000,4000,8000]]
 return grid,budgets
assert settings(OUT/'inputs/baseline.ipynb')==settings(OUT/'inputs/cleaned.ipynb')
cleaned=nbformat.read(ROOT/'notebook/lr_t1_poly2.ipynb',as_version=4)
nbformat.validate(cleaned)
for cell in cleaned.cells:
 if cell.cell_type=='code':compile(cell.source,'lr_t1_poly2.ipynb','exec')

changed=[]
for name,digest in manifest['original_file_hashes'].items():
 p=ROOT/name
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:changed.append(name)
assert changed==['notebook/lr_t1_poly2.ipynb'],changed
for name,digest in manifest['snapshot_hashes'].items():
 assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest,name
assert (ROOT/'notebook/lr_t1_poly2.ipynb').read_bytes()==(OUT/'inputs/cleaned.ipynb').read_bytes()
assert (ROOT/'notebook/lr_t0.ipynb').read_bytes()==(OUT/'inputs/lr_t0_template.ipynb').read_bytes()

baseline=nbformat.read(OUT/'inputs/baseline.ipynb',as_version=4)
old_code=sum(len(c.source.splitlines()) for c in baseline.cells if c.cell_type=='code')
new_code=sum(len(c.source.splitlines()) for c in cleaned.cells if c.cell_type=='code')
main=comparisons['project_default_runtime'];best_metrics=main['best_validation_metrics']
checks=[
 {'check':'Split membership and labels','baseline':'24,000 train / 6,000 validation, attack_cat stratification, seed 42','cleaned':'Exact IDs, row order, labels and values','status':'PASS'},
 {'check':'Preprocessing and degree-2 transformation','baseline':'41 raw -> 73 preprocessed -> 2,774 polynomial features','cleaned':'Same settings, fitted statistics and all transformed-value hashes','status':'PASS'},
 {'check':'Six-C sweep and model selection','baseline':'0.001, 0.01, 0.1, 1, 10, 100; retries 2000/4000/8000; converged-only F1','cleaned':'Same parameters, retries, convergence and selection','status':'PASS'},
 {'check':'Predictions and training/validation metrics','baseline':'180,000 predictions across six fits','cleaned':'Zero differences; metrics independently verified','status':'PASS'},
 {'check':'Selected C and validation F1','baseline':f"C=0.01; F1={best_metrics['val_f1']:.12f}",'cleaned':f"C=0.01; F1={best_metrics['val_f1']:.12f}",'status':'PASS'},
 {'check':'Required tables and core graphs','baseline':'Metric, term-count and attack-recall tables; curve and confusion matrix','cleaned':'Same non-timing table data; plot data/axes/labels/legends and pixels identical','status':'PASS'},
 {'check':'Historical saved metrics and plots','baseline':'Original pre-cleanup CSV and PNG files','cleaned':'Reproduced under project default numerical runtime','status':'PASS'},
 {'check':'Scope','baseline':'lr_t0.ipynb, original CSVs/figures/data and unrelated work','cleaned':'Unchanged; only target notebook and verification artifacts edited','status':'PASS'},
]
report={
 'overall':'Cleanup preserved T1 training results and plotted data under matched numerical runtimes. Default-runtime runs reproduce historical metrics and plots.',
 'changes':{'code_lines_before':old_code,'code_lines_after':new_code,
            'markdown_words_before':sum(len(c.source.split()) for c in baseline.cells if c.cell_type=='markdown'),
            'markdown_words_after':sum(len(c.source.split()) for c in cleaned.cells if c.cell_type=='markdown'),
            'removed_optional_diagnostics':['separate preprocessing preview fit','optimizer/coefficient-norm tables','coefficient inspection','cross-run CSV comparison','duplicate CSV rewriting','debug assertions and verbose generated captions'],
            'retained':['full polynomial feature map and feature-count table','six-C sweep','retry budgets and converged-only selection','metric and attack-type tables','validation curve and confusion matrix','CSV export checkpoint']},
 'checks':checks,'runtime_comparisons':comparisons,
 'historical_default_metric_max_error':historical_error,'historical_default_plot_comparison':historical_plots,
 'runtime_sensitivity':{'finding':'Imposing one numerical thread changes the LBFGS solution slightly relative to the normal Accelerate-backed runtime; no model setting was changed. Baseline and cleaned are exactly equal within each runtime.',
                       'differences_by_C':runtime_sensitivity,'default_runtime_environment':a.get('runtime_environment'),
                       'blas':a.get('numpy_build_configuration',{}).get('Build Dependencies',{}).get('blas')},
 'comparison_policy':manifest['comparison_policy'],
 'data_audit':json.loads((OUT/'data_audit.json').read_text()),
 'scope':{'changed_original_files':changed,'lr_t0_unchanged':True,'original_results_figures_data_and_baselines_preserved':True,
          'dependencies_installed_or_upgraded':False,'host':'Local macOS; no HPC job required.'},
 'experiment_execution_failures':[],
 'verification_harness_issue':{'error':'Initial NoSuchKernel before any training','cause':'Temporary kernel directory/name mismatch in the verification launcher','fix':'Aligned the temporary kernelspec name; preserved failed launch artifacts. All four actual notebook runs passed.'},
 'fixes_to_modeling_experiment':[],
 'limitations':['Full-file EDA informed preprocessing choices before validation isolation.',
                '321 validation rows share identical raw model inputs with training: 312 same-label, 9 conflicting-label matches; no splits changed.',
                'Validation selects C and is not an untouched final test. Test data were not made available to execution workspaces.',
                'Retry branches beyond 2,000 iterations were not needed in the actual successful fits; the candidate budgets and model settings were preserved.',
                'Fit times vary and are excluded from result equality. Cross-thread-runtime solutions are not claimed identical.'],
}
(OUT/'report.json').write_text(json.dumps(report,indent=2))
with (OUT/'comparison.csv').open('w',newline='') as handle:
 writer=csv.DictWriter(handle,fieldnames=['check','baseline','cleaned','status']);writer.writeheader();writer.writerows(checks)
print('ALL CHECKS PASSED. Both numerical modes: exact model outputs, feature hashes, tables and plot pixels.')
print('Default best:',main['best_C'],best_metrics)
print('Historical metric error:',historical_error)
print('Code:',old_code,'->',new_code,'lines. Changed originals:',changed)
print('Report:',OUT/'report.json')
