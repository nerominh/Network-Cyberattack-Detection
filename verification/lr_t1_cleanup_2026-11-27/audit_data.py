"""Read-only split regeneration and feature-overlap audit; no preprocessing fits."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
DATA=ROOT/'data'
raw=pd.read_csv(DATA/'UNSW_NB15_training_set.csv')
dedup=raw.drop_duplicates(subset=[c for c in raw.columns if c!='id'])
sample,_=train_test_split(dedup,train_size=30000,stratify=dedup['attack_cat'],random_state=42)
train,val=train_test_split(sample,test_size=6000,stratify=sample['attack_cat'],random_state=42)
saved_train=pd.read_csv(OUT/'snapshots/data/processed/train.csv')
saved_val=pd.read_csv(OUT/'snapshots/data/processed/val.csv')
assert np.array_equal(train['id'].to_numpy(),saved_train['id'].to_numpy())
assert np.array_equal(val['id'].to_numpy(),saved_val['id'].to_numpy())
assert np.array_equal(train['label'].to_numpy(),saved_train['label'].to_numpy())
assert np.array_equal(val['label'].to_numpy(),saved_val['label'].to_numpy())
assert set(train.id).isdisjoint(val.id)
for frame in [saved_train,saved_val]:
    assert np.array_equal(frame['label'].to_numpy(),(frame['attack_cat']!='Normal').astype(int).to_numpy())
excluded=['id','label','attack_cat','is_ftp_login']
features=saved_train.drop(columns=excluded).columns.tolist()
training=saved_train[features].copy()
training['_label']=saved_train.label
training=training.drop_duplicates()
validation=saved_val[features].copy()
validation['_label']=saved_val.label
validation['_valrow']=np.arange(len(validation))
shared=validation.merge(training,on=features,how='inner',suffixes=('_val','_train'))
same=shared[shared['_label_val']==shared['_label_train']]
conflict=shared[shared['_label_val']!=shared['_label_train']]
report={
 'raw_training_rows':len(raw),'deduplicated_rows':len(dedup),
 'sample_rows':len(sample),'training_rows':len(train),'validation_rows':len(val),
 'stratification':'attack_cat for both sample selection and train/validation split',
 'random_state':42,'regenerated_train_ids_and_order_match':True,'regenerated_val_ids_and_order_match':True,
 'source_id_overlap':0,'model_input_features':len(features),
 'attack_label_mapping_consistent':True,
 'shared_unique_feature_vectors':len(shared[features].drop_duplicates()),
 'validation_rows_with_a_training_feature_match':int(shared['_valrow'].nunique()),
 'validation_rows_with_same_feature_and_same_label_match':int(same['_valrow'].nunique()),
 'validation_rows_with_same_feature_and_conflicting_label_match':int(conflict['_valrow'].nunique()),
 'test_data_used':False,
 'preexisting_methodological_limitations':[
  'Full-training-file EDA informed fixed feature/category choices before validation was held out.',
  'Deduplication uses all columns except id, not model inputs alone; feature-vector overlap can remain across splits.',
  'Validation selects C and is therefore a model-selection score, not an untouched final-test estimate.',
 ]
}
overlap=saved_val.loc[sorted(shared['_valrow'].unique()),['id','label','attack_cat']].copy()
overlap['same_label_training_match']=overlap.index.isin(same['_valrow'])
overlap['conflicting_label_training_match']=overlap.index.isin(conflict['_valrow'])
overlap.to_csv(OUT/'overlapping_validation_rows.csv',index=False)
(OUT/'data_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
