import functools, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

EVENTS=[]
LOADS=[]
MODELS={}
PREDICTIONS={}
PLOTS={}
TEST_STARTED=False

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

def manual_scores(y,pred):
    y,pred=np.asarray(y),np.asarray(pred)
    tn=int(((y==0)&(pred==0)).sum());fp=int(((y==0)&(pred==1)).sum())
    fn=int(((y==1)&(pred==0)).sum());tp=int(((y==1)&(pred==1)).sum())
    return {'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn),
            'f1':2*tp/(2*tp+fp+fn),'fpr':fp/(fp+tn),'confusion':[[tn,fp],[fn,tp]]}

def install():
    global TEST_STARTED
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from matplotlib.figure import Figure
    import src.preprocessing as shared

    original_load=shared.load_split
    @functools.wraps(original_load)
    def load(name):
        global TEST_STARTED
        if name=='test':
            selected=pd.read_csv(shared.DATA.parent/'results/svm_t1_poly2_results.csv',keep_default_na=False)
            assert selected['selected'].sum()==1
            assert selected.loc[selected.selected,'converged'].all()
            TEST_STARTED=True
        result=original_load(name)
        LOADS.append({'name':name,'rows':len(result[0]),'features':len(result[0].columns),
                      'X':fingerprint(result[0]),'y_hash':sha_array(result[1].to_numpy())})
        return result
    shared.load_split=load

    for cls,method in [(ColumnTransformer,'fit'),(ColumnTransformer,'fit_transform'),(StandardScaler,'fit'),(LinearSVC,'fit')]:
        original=getattr(cls,method)
        def wrap(original,cls,method):
            @functools.wraps(original)
            def fit(self,X,y=None,*args,**kwargs):
                if TEST_STARTED:raise AssertionError('Fitting after test loading is prohibited by this protocol.')
                assert len(X)==24000
                event={'class':cls.__name__,'method':method,'shape':list(X.shape),
                       'y_hash':None if y is None else sha_array(np.asarray(y))}
                if isinstance(X,pd.DataFrame):event['X']=fingerprint(X)
                if cls is LinearSVC:
                    assert X.shape[1] in {73,2774} and np.isfinite(X).all()
                    event['C']=self.C;event['estimator_id']=id(self)
                EVENTS.append(event)
                result=original(self,X,y,*args,**kwargs)
                if cls is LinearSVC:MODELS[id(self)]=self
                return result
            return fit
        setattr(cls,method,wrap(original,cls,method))

    original_predict=LinearSVC.predict
    @functools.wraps(original_predict)
    def predict(self,X):
        assert np.isfinite(X).all()
        pred=original_predict(self,X)
        PREDICTIONS.setdefault(id(self),[]).append({'test':TEST_STARTED,'rows':len(X),'pred':np.asarray(pred).copy()})
        return pred
    LinearSVC.predict=predict
    original_save=Figure.savefig
    @functools.wraps(original_save)
    def save(self,fname,*args,**kwargs):
        result=original_save(self,fname,*args,**kwargs)
        if isinstance(fname,(str,Path)):PLOTS[Path(fname).name]=plot_snapshot(self,kwargs)
        return result
    Figure.savefig=save


def capture(namespace):
    out=Path(__file__).resolve().parent
    results=namespace['results'];final=namespace['final_results'];best=namespace['best'];selected=namespace['best_model']
    y_train=namespace['y_train'].to_numpy();y_val=namespace['y_val'].to_numpy();y_test=namespace['y_test'].to_numpy()
    arrays={'y_train':y_train,'y_val':y_val,'y_test':y_test,'pred_test':namespace['pred_test'],'pred_val':namespace['pred_val']}
    report={'selection_C':float(best['C']),'results':results.to_dict(orient='records'),
            'final':final.to_dict(orient='records'),'load_events':LOADS,'fit_events':EVENTS,'plots':PLOTS,
            'independent_metrics':[]}
    assert results.C.tolist()==[.001,.01,.1,1,10,100]
    assert len(results)==6 and results.selected.sum()==1
    assert results.loc[results.selected,'converged'].all()
    assert float(results.loc[results.converged].sort_values('val_f1',ascending=False,kind='stable').iloc[0].C)==float(best.C)
    assert all(e['rows'] in {24000,6000,82332} for e in LOADS)
    assert [e['name'] for e in LOADS]==['train','val','test']
    assert len(final)==1 and final.iloc[0]['C']==best.C and not final.iloc[0]['final_refit']
    for i,C in enumerate(namespace['GRID']):
        pipe=namespace['fitted'][C];classifier=pipe[-1]
        assert len(pipe.steps)==4 and classifier.n_features_in_==2774
        assert pipe.named_steps['polynomialfeatures'].n_output_features_==2774
        for scale in [pipe.named_steps['standardscaler'],pipe[0].named_transformers_['log'].named_steps['standardscaler'],pipe[0].named_transformers_['num']]:
            assert scale.n_samples_seen_==24000
        calls=[call for call in PREDICTIONS[id(classifier)] if not call['test']]
        predictions=np.concatenate([call['pred'] for call in calls])
        assert len(predictions)>=30000
        train_pred,val_pred=predictions[:24000],predictions[24000:30000]
        arrays[f'c{i}_train_pred']=train_pred;arrays[f'c{i}_val_pred']=val_pred
        row=results.loc[results.C==C].iloc[0]
        for split,y,pred in [('train',y_train,train_pred),('val',y_val,val_pred)]:
            independent=manual_scores(y,pred)
            for metric in ['precision','recall','f1','fpr']:
                assert abs(row[f'{split}_{metric}']-independent[metric])<=1e-15
            assert independent['confusion']==[[row[f'{split}_tn'],row[f'{split}_fp']],[row[f'{split}_fn'],row[f'{split}_tp']]]
            report['independent_metrics'].append({'C':C,'split':split,**independent})
    test_calls=[(model_id,call) for model_id,calls in PREDICTIONS.items() for call in calls if call['test']]
    assert {model_id for model_id,_ in test_calls}=={id(selected[-1])}
    assert sum(call['rows'] for _,call in test_calls)==82332
    assert len(test_calls)==int(np.ceil(82332/namespace['BATCH_SIZE']))
    assert np.array_equal(np.concatenate([call['pred'] for _,call in test_calls]),namespace['pred_test'])
    manual_test=manual_scores(y_test,namespace['pred_test'])
    for metric in ['precision','recall','f1','fpr']:
        assert abs(final.iloc[0][f'test_{metric}']-manual_test[metric])<=1e-15
    report['independent_final_test_metrics']=manual_test
    report['test_prediction_batches']=len(test_calls)
    report['test_predictions_only_selected_model']=True
    report['all_fits_on_24000_training_rows_before_test']=True
    train_fingerprint=next(e['X'] for e in LOADS if e['name']=='train')
    train_yhash=sha_array(y_train)
    for event in EVENTS:
        if event['class']=='ColumnTransformer':assert event['X']==train_fingerprint
        if event['class']=='LinearSVC':assert event['y_hash']==train_yhash
    # Independently construct all polynomial terms on a few training rows.
    from sklearn.preprocessing import PolynomialFeatures
    U=selected[0].transform(namespace['X_train'].iloc[:3]);poly=selected.named_steps['polynomialfeatures']
    powers=poly.powers_;actual=poly.transform(U);columns=[]
    for power in powers:
        factors=np.flatnonzero(power)
        if len(factors)==1:
            column=U[:,factors[0]].copy()
            if power[factors[0]]==2:column=column*column
        else:
            assert len(factors)==2 and (power[factors]==1).all()
            column=U[:,factors[0]]*U[:,factors[1]]
        columns.append(column)
    manual_poly=np.column_stack(columns)
    assert np.array_equal(manual_poly,actual)
    assert np.array_equal(selected.named_steps['standardscaler'].transform(actual),selected[:-1].transform(namespace['X_train'].iloc[:3]))
    report['independent_polynomial_term_check']='Exact equality for all 2,774 terms on three training examples.'
    # Same effective estimator settings as SVM T0, and transformation as LR T1.
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from src.preprocessing import build_preprocessor,RANDOM_STATE
    assert selected[-1].get_params()==LinearSVC(C=best.C,dual='auto',max_iter=int(best.max_iter_used),random_state=RANDOM_STATE).get_params()
    assert poly.get_params()==PolynomialFeatures(degree=2,include_bias=False,interaction_only=False).get_params()
    assert selected.named_steps['standardscaler'].get_params()==StandardScaler().get_params()
    curve=PLOTS['curve_svm_t1_poly2.png']['axes']
    for axis,metrics in [(curve[0],['train_f1','val_f1']),(curve[1],['val_precision','val_recall','val_fpr'])]:
        assert axis['xscale']=='log'
        for line,metric in zip(axis['lines'],metrics):
            assert np.array_equal(line['x'],results.value.to_numpy()) and np.array_equal(line['y'],results[metric].to_numpy())
    for name,y,pred in [('confusion_svm_t1_poly2_validation.png',y_val,namespace['pred_val']),('confusion_svm_t1_poly2_test.png',y_test,namespace['pred_test'])]:
        axis=PLOTS[name]['axes'][0]
        assert axis['xticklabels']==axis['yticklabels']==['Normal','Attack']
        assert np.array_equal(axis['images'][0]['array'],manual_scores(y,pred)['confusion'])
    environment={}
    import importlib.metadata,sys,platform
    environment.update(python=sys.version,executable=sys.executable,host=platform.platform(),packages={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','matplotlib','nbformat','nbclient','ipykernel']})
    report['environment']=environment
    np.savez_compressed(out/'predictions.npz',**arrays)
    (out/'execution_audit.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False))
    print('AUDIT PASS: independent candidate/test metrics, polynomial values, plot data, training-only fits and selected-only test evaluation.')
