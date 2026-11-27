"""Read-only notebook instrumentation; no fit inputs or predictions are modified."""
import functools
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIT_EVENTS=[]
LOAD_EVENTS=[]
PLOTS={}

def sha_array(array):
    array=np.asarray(array)
    header=json.dumps({'shape':list(array.shape),'dtype':str(array.dtype)},sort_keys=True).encode()
    return hashlib.sha256(header+np.ascontiguousarray(array).tobytes()).hexdigest()

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
    from sklearn.preprocessing import StandardScaler
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
                       (StandardScaler,'fit'),(LogisticRegression,'fit'),(LinearSVC,'fit')]:
        original=getattr(cls,method)
        def make_wrapper(original,cls,method):
            @functools.wraps(original)
            def wrapper(self,X,y=None,*args,**kwargs):
                FIT_EVENTS.append({'class':cls.__name__,'method':method,'X':fingerprint(X),
                                   'y':None if y is None else sha_array(np.asarray(y))})
                return original(self,X,y,*args,**kwargs)
            return wrapper
        setattr(cls,method,make_wrapper(original,cls,method))

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


def capture(namespace):
    work=Path.cwd().parent
    destination=work.parent/'capture'
    destination.mkdir(exist_ok=True)
    X_train,X_val=namespace['X_train'],namespace['X_val']
    y_train,y_val=namespace['y_train'].to_numpy(),namespace['y_val'].to_numpy()
    results=namespace['results']
    arrays={'y_train':y_train,'y_val':y_val,
            'train_index':X_train.index.to_numpy(),'val_index':X_val.index.to_numpy(),
            'train_ids':pd.read_csv(work/'data/processed/train.csv',usecols=['id'])['id'].to_numpy(),
            'val_ids':pd.read_csv(work/'data/processed/val.csv',usecols=['id'])['id'].to_numpy()}
    report={'raw_X_train':fingerprint(X_train),'raw_X_val':fingerprint(X_val),
            'best_C':float(namespace['best']['value']),
            'grid':[float(v) for v in namespace['GRID']],
            'results':results.to_dict(orient='records'),
            'fit_events':FIT_EVENTS,'load_events':LOAD_EVENTS,'plots':PLOTS,
            'models':[], 'metric_tolerance':1e-15,
            'metric_averaging':'binary, attack label 1; FPR conditions on label 0'}
    assert set(np.unique(y_train))=={0,1} and set(np.unique(y_val))=={0,1}
    assert X_train.index.equals(namespace['y_train'].index)
    assert X_val.index.equals(namespace['y_val'].index)
    assert not set(arrays['train_ids']).intersection(arrays['val_ids'])
    assert not {'id','label','attack_cat','is_ftp_login'}.intersection(X_train.columns)
    for i,C in enumerate(namespace['GRID']):
        pipe=namespace['fitted'][C]
        prep=pipe.named_steps['columntransformer']
        classifier=pipe.steps[-1][1]
        train_transformed=prep.transform(X_train)
        val_transformed=prep.transform(X_val)
        assert train_transformed.shape==(24000,73) and val_transformed.shape==(6000,73)
        assert np.isfinite(train_transformed).all() and np.isfinite(val_transformed).all()
        if i==0:
            arrays['train_features']=train_transformed
            arrays['val_features']=val_transformed
        else:
            assert np.array_equal(arrays['train_features'],train_transformed)
            assert np.array_equal(arrays['val_features'],val_transformed)
        entry={'C':float(C),'estimator_class':type(classifier).__name__,
               'estimator_params':classifier.get_params(),
               'preprocessor_params':prep.get_params(deep=False),
               'feature_names':prep.get_feature_names_out().tolist(),
               'scalers':{},'independent_scores':{}}
        for label,scale in [('log',prep.named_transformers_['log'].named_steps['standardscaler']),
                            ('num',prep.named_transformers_['num'])]:
            entry['scalers'][label]={name:getattr(scale,name) for name in ['mean_','scale_','var_','n_samples_seen_']}
            assert scale.n_samples_seen_==24000
        entry['encoder_categories']=prep.named_transformers_['cat'].named_steps['onehotencoder'].categories_
        arrays[f'c{i}_coef']=classifier.coef_
        arrays[f'c{i}_intercept']=classifier.intercept_
        arrays[f'c{i}_classes']=classifier.classes_
        arrays[f'c{i}_iterations']=np.asarray(classifier.n_iter_)
        row=results.loc[results['value']==C].iloc[0]
        for label,X,y in [('train',X_train,y_train),('val',X_val,y_val)]:
            pred=pipe.predict(X)
            scores=pipe.decision_function(X)
            arrays[f'c{i}_{label}_predictions']=pred
            arrays[f'c{i}_{label}_scores']=scores
            assert np.array_equal((scores>0).astype(int),pred)
            if hasattr(pipe,'predict_proba'):
                arrays[f'c{i}_{label}_probabilities']=pipe.predict_proba(X)
            independent=manual_scores(y,pred)
            entry['independent_scores'][label]=independent
            for metric in ['precision','recall','f1','fpr']:
                assert abs(float(row[f'{label}_{metric}'])-independent[metric])<=1e-15,(C,label,metric)
        assert abs(row['f1_gap']-(row['train_f1']-row['val_f1']))<=1e-15
        report['models'].append(entry)
    assert all(e['name'] in {'train','val'} for e in LOAD_EVENTS)
    train_fingerprint=fingerprint(X_train)
    for event in FIT_EVENTS:
        assert event['X']['shape'][0]==24000,event
        if event['class']=='ColumnTransformer':
            assert event['X']==train_fingerprint,event
        if event['class'] in {'LogisticRegression','LinearSVC'}:
            assert event['y']==sha_array(y_train)
            assert event['X']['hash']==sha_array(arrays['train_features'])
    eligible=results[results['converged']]
    expected_best=float(eligible.loc[eligible.val_f1.idxmax(),'value'])
    assert expected_best==report['best_C']
    best_index=report['grid'].index(report['best_C'])
    assert np.array_equal(namespace['pred_val'],arrays[f'c{best_index}_val_predictions'])
    pred=arrays[f'c{best_index}_val_predictions']
    categories=namespace['cat_val'].to_numpy()
    attack_stats=[]
    for category in sorted(set(categories[y_val==1])):
        mask=(y_val==1)&(categories==category)
        rows=int(mask.sum());caught=int(pred[mask].sum())
        attack_stats.append({'type':category,'rows':rows,'caught':caught,'missed':rows-caught,'recall':round(caught/rows,3)})
    report['attack_type_recall']=attack_stats
    # Plot data checked independently against metrics and the best predictions.
    curve_name=next(name for name in PLOTS if name.startswith('curve_'))
    curve=PLOTS[curve_name]['axes']
    for axis,metrics in [(curve[0],['train_f1','val_f1']),(curve[1],['val_precision','val_recall','val_fpr'])]:
        assert axis['xscale']=='log'
        for line,metric in zip(axis['lines'],metrics):
            assert np.array_equal(np.asarray(line['x']),results.value.to_numpy())
            assert np.array_equal(np.asarray(line['y']),results[metric].to_numpy())
    confusion_name=next(name for name in PLOTS if name.startswith('confusion_'))
    axis=PLOTS[confusion_name]['axes'][0]
    assert axis['xticklabels']==['Normal','Attack'] and axis['yticklabels']==['Normal','Attack']
    assert np.array_equal(np.asarray(axis['images'][0]['array']),manual_scores(y_val,pred)['confusion'])
    np.savez_compressed(destination/'arrays.npz',**arrays)
    (destination/'run.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False))
    print(f'VERIFIED {len(report["models"])} models; independent metrics and plot data agree. Best C={report["best_C"]:g}.')
