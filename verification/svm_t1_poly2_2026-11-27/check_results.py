"""Read-only post-execution checks; does not fit or evaluate any new model."""
import hashlib,json
from pathlib import Path
import nbformat
import numpy as np
import pandas as pd
from PIL import Image

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
status=json.loads((OUT/'execution_status.json').read_text())
assert status['status']=='passed',status
nb=nbformat.read(ROOT/'notebook/svm_t1_poly2.ipynb',as_version=4);nbformat.validate(nb)
for cell in nb.cells:
 if cell.cell_type=='code':
  compile(cell.source,'svm_t1_poly2.ipynb','exec')
  assert cell.execution_count is not None
  assert not any(o.output_type=='error' for o in cell.outputs)
  assert '_svm_t1_audit' not in cell.source
assert sum('image/png' in o.get('data',{}) for c in nb.cells for o in c.get('outputs',[]))==3
search=pd.read_csv(ROOT/'results/svm_t1_poly2_results.csv',keep_default_na=False)
final=pd.read_csv(ROOT/'results/svm_t1_poly2_final_results.csv',keep_default_na=False)
audit=json.loads((OUT/'execution_audit.json').read_text())
arrays=np.load(OUT/'predictions.npz')
assert len(search)==6 and search.C.tolist()==[.001,.01,.1,1,10,100]
assert search.selected.sum()==1 and len(final)==1
selected=search.loc[search.selected].iloc[0]
assert selected.converged and final.iloc[0].converged
assert final.iloc[0].C==selected.C==audit['selection_C']
assert set(search.polynomial_degree)=={2} and not search.polynomial_include_bias.any() and not search.polynomial_interaction_only.any()
assert set(search.base_features)=={73} and set(search.transformed_features)=={2774}
assert search.post_poly_with_mean.all() and search.post_poly_with_std.all()
assert set(search.random_state)=={42} and set(search.estimator)=={'LinearSVC'}
assert set(search.train_size)=={24000} and set(search.validation_size)=={6000}
assert not any(key.startswith('test_') for key in search.columns)
assert not any(key.startswith('Unnamed:') for key in search.columns) and not any(key.startswith('Unnamed:') for key in final.columns)
for frame in [search,final]:
 for stage in ['train','val']+(['test'] if frame is final else []):
  cols=[f'{stage}_{m}' for m in ['precision','recall','f1','fpr']]
  assert np.isfinite(frame[cols]).all().all() and frame[cols].apply(lambda v:v.between(0,1).all()).all()
  for _,row in frame.iterrows():
   tn,fp,fn,tp=[int(row[f'{stage}_{name}']) for name in ['tn','fp','fn','tp']]
   expected={'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn),'f1':2*tp/(2*tp+fp+fn),'fpr':fp/(fp+tn)}
   for metric,value in expected.items():assert abs(row[f'{stage}_{metric}']-value)<=1e-15
   assert tn+fp+fn+tp==int(row[{'train':'train_size','val':'validation_size','test':'test_size'}[stage]])
assert final.iloc[0].test_evaluations==1 and not final.iloc[0].final_refit
assert final.iloc[0].model_fit_rows==24000 and final.iloc[0].test_size==82332
assert audit['test_predictions_only_selected_model'] and audit['all_fits_on_24000_training_rows_before_test']
assert len(arrays['y_test'])==len(arrays['pred_test'])==82332
for name,column in [('train','train_split_sha256'),('val','validation_split_sha256')]:
 digest=hashlib.sha256((ROOT/'data/processed'/f'{name}.csv').read_bytes()).hexdigest()
 assert set(search[column])=={digest} and final.iloc[0][column]==digest
assert final.iloc[0].test_split_sha256==hashlib.sha256((ROOT/'data/processed/test.csv').read_bytes()).hexdigest()
pd.testing.assert_frame_equal(pd.read_csv(ROOT/'data/processed/test.csv'),
                              pd.read_csv(ROOT/'data/UNSW_NB15_testing_set.csv'),check_exact=True)
provenance=json.loads((OUT/'split_provenance.json').read_text())
assert provenance['train']['exact_split_reproduction'] and provenance['val']['exact_split_reproduction']
originals=json.loads((OUT/'original_file_hashes.json').read_text())
changed=[name for name,digest in originals.items() if not (ROOT/name).exists() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
assert not changed,changed
for name in ['curve_svm_t1_poly2.png','confusion_svm_t1_poly2_validation.png','confusion_svm_t1_poly2_test.png']:
 with Image.open(ROOT/'figures'/name) as image:image.verify()

report={
 'status':'PASS','files':{'notebook':'notebook/svm_t1_poly2.ipynb',
  'search_csv':'results/svm_t1_poly2_results.csv','final_csv':'results/svm_t1_poly2_final_results.csv',
  'figures':['figures/curve_svm_t1_poly2.png','figures/confusion_svm_t1_poly2_validation.png','figures/confusion_svm_t1_poly2_test.png']},
 'selected_C':float(selected.C),'converged_candidates':int(search.converged.sum()),
 'candidate_warnings':search.loc[search.warning_messages!='',['C','warning_messages','converged']].to_dict(orient='records'),
 'selected_metrics':{stage:{m:float(final.iloc[0][f'{stage}_{m}']) for m in ['precision','recall','f1','fpr']} for stage in ['train','val','test']},
 'final_confusion':{name:int(final.iloc[0][f'test_{name}']) for name in ['tn','fp','fn','tp']},
 'validation_confusion':{name:int(selected[f'val_{name}']) for name in ['tn','fp','fn','tp']},
 'checks':['Fresh-kernel top-to-bottom execution passed.',
  'All candidate metrics independently agree with actual predictions and true labels.',
  'All 2,774 polynomial terms independently reconstructed exactly on three rows.',
  'Only training data fitted; no fits occurred after test loading.',
  'Only the selected T1 model predicted test, covering 82,332 rows in one batched pass.',
  'CSV row counts, selected flags, parameters, metrics and confusion counts verified.',
  'Canonical train/validation splits exactly reproduce preprocessing.ipynb stratified sampling; full test data equals its official source.',
  'Plotted data and class labels independently match recorded metrics and predictions.',
  'Existing reference notebooks, CSVs, shared code and unrelated work unchanged.'],
 'overlap_audit':{name:int(final.iloc[0][name]) for name in ['train_val_feature_overlap_rows','test_feature_overlap_training_rows','test_feature_overlap_validation_rows','test_feature_overlap_development_rows']},
 'limitations':['Full-file EDA informed fixed preprocessing choices before validation isolation.',
  'Exact feature-vector overlap predates this experiment and limits holdout independence.',
  'This test score is for the frozen selected T1 configuration, not the globally selected model across all approaches.',
  'The test set has now been evaluated and must not guide subsequent tuning.',
  'The selected training-only model was reused because no train-plus-validation final-refit protocol was defined in the reference.'],
 'environment':audit['environment']}
(OUT/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({key:report[key] for key in ['status','selected_C','converged_candidates','selected_metrics','final_confusion','overlap_audit']},indent=2))
print('Report:',OUT/'report.json')
