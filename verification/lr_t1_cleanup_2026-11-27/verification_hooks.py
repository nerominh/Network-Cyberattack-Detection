"""Read-only notebook instrumentation; no fit inputs or predictions are modified."""
import os
import functools
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIT_EVENTS=[]
LOAD_EVENTS=[]
PLOTS={}
OBSERVATIONS={}

def sha_array(array):
    array=np.asarray(array)
    header=json.dumps({'shape':list(array.shape),'dtype':str(array.dtype)},sort_keys=True).encode()
    digest=hashlib.sha256(header)
    digest.update(memoryview(np.ascontiguousarray(array)).cast('B'))
    return digest.hexdigest()

def fingerprint(value):
    if isinstance(value,pd.DataFrame):
        return {'shape':list(value.shape),'columns':list(value.columns),
                'dtypes':[str(x) for x in value.dtypes],
                'hash':hashlib.sha256(pd.util.hash_pandas_object(value,index=True).values.tobytes()).hexdigest()}
    return {'shape':list(np.asarray(value).shape),'hash':sha_array(value)}

def clean(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if hasattr(value,'get_params'):
        return {'class':type(value).__module__+'.'+type(value).__name__,'params':clean(value.get_params(deep=False))}
    if callable(value):return getattr(value,'__module__',type(value).__module__)+'.'+getattr(value,'__qualname__',getattr(value,'__name__',type(value).__name__))
    return str(value)

def plot_snapshot(fig,kwargs):
    axes=[]
    for ax in fig.axes:
        legend=ax.get_legend()
        axes.append({
            'title':ax.get_title(),'xlabel':ax.get_xlabel(),'ylabel':ax.get_ylabel(),
            'xscale':ax.get_xscale(),'yscale':ax.get_yscale(),
            'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim()),
            'xticks':ax.get_xticks().tolist(),'yticks':ax.get_yticks().tolist(),
            'xticklabels':[t.get_text() for t in ax.get_xticklabels()],
            'yticklabels':[t.get_text() for t in ax.get_yticklabels()],
            'position':list(ax.get_position().bounds),
            'legend':None if legend is None else [t.get_text() for t in legend.get_texts()],
            'lines':[{'x':clean(np.asarray(line.get_xdata())),'y':clean(np.asarray(line.get_ydata())),
                      'label':line.get_label(),'color':clean(line.get_color()),'linestyle':line.get_linestyle(),
                      'marker':line.get_marker(),'linewidth':line.get_linewidth(),'markersize':line.get_markersize()}
                     for line in ax.lines],
            'images':[{'array':clean(np.asarray(im.get_array())),'cmap':im.get_cmap().name,
                       'extent':clean(im.get_extent()),'clim':clean(im.get_clim())} for im in ax.images],
            'texts':[{'text':t.get_text(),'position':clean(t.get_position()),'color':clean(t.get_color()),
                      'fontsize':t.get_fontsize(),'rotation':t.get_rotation()} for t in ax.texts],
        })
    return clean({'size_inches':fig.get_size_inches(),'figure_dpi':fig.dpi,'save_kwargs':kwargs,'axes':axes})

def install_hooks():
    from matplotlib.figure import Figure
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    import src.preprocessing as shared

    original_load=shared.load_split
    @functools.wraps(original_load)
    def load(name):
        if name not in {'train','val'}:raise RuntimeError('Verification prohibits test-set loading.')
        result=original_load(name)
        LOAD_EVENTS.append({'name':name,'X':fingerprint(result[0]),'y':sha_array(result[1].to_numpy())})
        return result
    shared.load_split=load

    for cls,method in [(ColumnTransformer,'fit'),(ColumnTransformer,'fit_transform'),
                       (StandardScaler,'fit'),(PolynomialFeatures,'fit'),(LogisticRegression,'fit'),(LinearSVC,'fit')]:
        original=getattr(cls,method)
        def make_wrapper(original,cls,method):
            @functools.wraps(original)
            def wrapper(self,X,y=None,*args,**kwargs):
                FIT_EVENTS.append({'class':cls.__name__,'method':method,'estimator_id':id(self),'X':fingerprint(X),
                                   'y':None if y is None else sha_array(np.asarray(y))})
                return original(self,X,y,*args,**kwargs)
            return wrapper
        setattr(cls,method,make_wrapper(original,cls,method))

    # Observe predictions actually used by evaluate(); do not transform the
    # entire 2,774-column training matrix repeatedly just to capture results.
    original_scores=LogisticRegression.decision_function
    @functools.wraps(original_scores)
    def decision(self,X):
        result=original_scores(self,X)
        item=OBSERVATIONS.setdefault(id(self),{'estimator':self,'splits':{}})
        split=item['splits'].setdefault(len(X),{})
        if 'feature_hash' not in split:split['feature_hash']=sha_array(X)
        split['scores']=np.asarray(result).copy()
        return result
    LogisticRegression.decision_function=decision
    original_predict=LogisticRegression.predict
    @functools.wraps(original_predict)
    def predict(self,X):
        result=original_predict(self,X)
        item=OBSERVATIONS.setdefault(id(self),{'estimator':self,'splits':{}})
        item['splits'].setdefault(len(X),{})['predictions']=np.asarray(result).copy()
        return result
    LogisticRegression.predict=predict

    original_save=Figure.savefig
    @functools.wraps(original_save)
    def save(fig,fname,*args,**kwargs):
        result=original_save(fig,fname,*args,**kwargs)
        if isinstance(fname,(str,Path)):
            PLOTS[Path(fname).name]=plot_snapshot(fig,kwargs)
        return result
    Figure.savefig=save


def manual_scores(y,pred):
    y,pred=np.asarray(y),np.asarray(pred)
    tn=int(((y==0)&(pred==0)).sum());fp=int(((y==0)&(pred==1)).sum())
    fn=int(((y==1)&(pred==0)).sum());tp=int(((y==1)&(pred==1)).sum())
    return {'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn),
            'f1':2*tp/(2*tp+fp+fn),'fpr':fp/(fp+tn),'confusion':[[tn,fp],[fn,tp]]}


def chunk_hash(transformer,X,features):
    """Hash every transformed float in row order without storing a 0.5-GiB copy."""
    header=json.dumps({'shape':[len(X),features],'dtype':'float64'},sort_keys=True).encode()
    digest=hashlib.sha256(header)
    for start in range(0,len(X),512):
        result=np.ascontiguousarray(transformer.transform(X.iloc[start:start+512]))
        assert result.shape==(min(512,len(X)-start),features)
        assert result.dtype==np.float64 and np.isfinite(result).all()
        digest.update(memoryview(result).cast('B'))
    return digest.hexdigest()


def capture(namespace):
    work=Path.cwd().parent;destination=work.parent/'capture';destination.mkdir(exist_ok=True)
    X_train,X_val=namespace['X_train'],namespace['X_val']
    y_train,y_val=namespace['y_train'].to_numpy(),namespace['y_val'].to_numpy()
    results=namespace['results'];fitted=namespace['fitted']
    assert X_train.shape==(24000,41) and X_val.shape==(6000,41)
    assert X_train.columns.equals(X_val.columns)
    assert X_train.index.equals(namespace['y_train'].index) and X_val.index.equals(namespace['y_val'].index)
    assert not {'id','label','attack_cat','is_ftp_login'}.intersection(X_train.columns)
    assert set(np.unique(y_train))=={0,1} and set(np.unique(y_val))=={0,1}
    arrays={'y_train':y_train,'y_val':y_val,
            'train_ids':pd.read_csv(work/'data/processed/train.csv',usecols=['id'])['id'].to_numpy(),
            'val_ids':pd.read_csv(work/'data/processed/val.csv',usecols=['id'])['id'].to_numpy(),
            'train_index':X_train.index.to_numpy(),'val_index':X_val.index.to_numpy()}
    assert set(arrays['train_ids']).isdisjoint(arrays['val_ids'])
    best_C=float(namespace['best']['value']);best_pipe=fitted[best_C]
    prep=best_pipe.named_steps['columntransformer'];poly=best_pipe.named_steps['polynomialfeatures']
    arrays['train_preprocessed']=prep.transform(X_train)
    arrays['val_preprocessed']=prep.transform(X_val)
    arrays['polynomial_powers']=poly.powers_
    assert arrays['train_preprocessed'].shape==(24000,73)
    assert arrays['val_preprocessed'].shape==(6000,73)
    assert poly.n_output_features_==2774
    powers=poly.powers_;names=poly.get_feature_names_out(prep.get_feature_names_out())
    masks=[('Linear',powers.sum(axis=1)==1),('Squares',(powers==2).any(axis=1)),
           ('Pairwise products',(powers==1).sum(axis=1)==2)]
    terms=[{'term_type':name,'columns':int(mask.sum()),'examples':list(names[mask][:3])} for name,mask in masks]
    assert [r['columns'] for r in terms]==[73,73,2628]
    report={'raw_X_train':fingerprint(X_train),'raw_X_val':fingerprint(X_val),
            'grid':[float(x) for x in namespace['GRID']],'best_C':best_C,
            'results':results.to_dict(orient='records'),'models':[],
            'fit_events':FIT_EVENTS,'load_events':LOAD_EVENTS,'plots':PLOTS,
            'feature_names':names.tolist(),'term_table':terms,
            'transformation_hashes':{
                'train_polynomial_unscaled':chunk_hash(best_pipe[:2],X_train,2774),
                'val_polynomial_unscaled':chunk_hash(best_pipe[:2],X_val,2774),
                'train_polynomial_scaled':chunk_hash(best_pipe[:-1],X_train,2774),
                'val_polynomial_scaled':chunk_hash(best_pipe[:-1],X_val,2774)},
            'metric_averaging':'binary attack=1; FPR conditions on normal=0',
            'runtime_environment':{key:os.environ.get(key) for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']},
            'numpy_build_configuration':getattr(np.__config__,'CONFIG',None)}
    training_fingerprint=fingerprint(X_train)
    for event in FIT_EVENTS:
        if event['class']=='PolynomialFeatures':
            assert event['X']['shape'][0] in {1,24000} and event['X']['shape'][1]==73
            if event['X']['shape'][0]==1:
                assert event['X']['hash']==sha_array(arrays['train_preprocessed'][:1])
        else:assert event['X']['shape'][0]==24000
        if event['class']=='ColumnTransformer':assert event['X']==training_fingerprint
        if event['class']=='LogisticRegression':
            assert event['y']==sha_array(y_train)
            assert event['X']['shape']==[24000,2774]
            assert event['X']['hash']==report['transformation_hashes']['train_polynomial_scaled']
    assert {e['name'] for e in LOAD_EVENTS}=={'train','val'}
    for i,C in enumerate(namespace['GRID']):
        pipe=fitted[C];classifier=pipe.named_steps['logisticregression'];current=pipe.named_steps['columntransformer']
        assert classifier.n_features_in_==2774
        entry={'C':float(C),'estimator_params':classifier.get_params(),
               'pipeline_configuration':pipe.get_params(deep=False),
               'scalers':{},'independent_scores':{}}
        scales={'log':current.named_transformers_['log'].named_steps['standardscaler'],
                'num':current.named_transformers_['num'],
                'polynomial':pipe.named_steps['standardscaler']}
        for label,scale in scales.items():
            assert scale.n_samples_seen_==24000
            entry['scalers'][label]={key:getattr(scale,key) for key in ['mean_','var_','scale_','n_samples_seen_']}
        arrays[f'c{i}_coef']=classifier.coef_;arrays[f'c{i}_intercept']=classifier.intercept_
        arrays[f'c{i}_classes']=classifier.classes_;arrays[f'c{i}_iterations']=np.asarray(classifier.n_iter_)
        row=results.loc[results.value==C].iloc[0]
        observed=OBSERVATIONS[id(classifier)]['splits']
        for label,y in [('train',y_train),('val',y_val)]:
            data=observed[len(y)];pred=data['predictions'];scores=data['scores']
            assert data['feature_hash']==report['transformation_hashes'][f'{label}_polynomial_scaled']
            arrays[f'c{i}_{label}_predictions']=pred;arrays[f'c{i}_{label}_scores']=scores
            assert np.array_equal((scores>0).astype(int),pred)
            independent=manual_scores(y,pred);entry['independent_scores'][label]=independent
            for metric in ['precision','recall','f1','fpr']:
                assert abs(float(row[f'{label}_{metric}'])-independent[metric])<=1e-15
        assert abs(row['f1_gap']-row['train_f1']+row['val_f1'])<=1e-15
        report['models'].append(entry)
    eligible=results[results.converged]
    assert float(eligible.loc[eligible.val_f1.idxmax(),'value'])==best_C
    best_i=report['grid'].index(best_C);pred=arrays[f'c{best_i}_val_predictions']
    assert np.array_equal(namespace['pred_val'],pred)
    categories=namespace['cat_val'].to_numpy();attack_stats=[]
    for category in sorted(set(categories[y_val==1])):
        mask=(y_val==1)&(categories==category);rows=int(mask.sum());caught=int(pred[mask].sum())
        attack_stats.append({'type':category,'rows':rows,'caught':caught,'missed':rows-caught,'recall':round(caught/rows,3)})
    report['attack_type_recall']=attack_stats
    curve=PLOTS['curve_lr_t1_poly2.png']['axes']
    for ax,metrics in [(curve[0],['train_f1','val_f1']),(curve[1],['val_precision','val_recall','val_fpr'])]:
        assert ax['xscale']=='log'
        for line,metric in zip(ax['lines'],metrics):
            assert np.array_equal(np.asarray(line['x']),results.value.to_numpy())
            assert np.array_equal(np.asarray(line['y']),results[metric].to_numpy())
    confusion=PLOTS['confusion_lr_t1_poly2.png']['axes'][0]
    assert confusion['xticklabels']==confusion['yticklabels']==['Normal','Attack']
    assert np.array_equal(confusion['images'][0]['array'],manual_scores(y_val,pred)['confusion'])
    np.savez_compressed(destination/'arrays.npz',**arrays)
    (destination/'run.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False))
    print(f'VERIFIED six fits, all transformed values by streaming hashes, independent metrics and core plot data; best C={best_C:g}.')
