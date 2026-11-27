import ast,builtins,json,symtable
from pathlib import Path
import nbformat
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.linear_model import LogisticRegression
from src.preprocessing import RANDOM_STATE,build_preprocessor
out=Path('verification/lr_t1_cleanup_2026-11-27')
a=nbformat.read(out/'inputs/baseline.ipynb',as_version=4)
b=nbformat.read('notebook/lr_t1_poly2.ipynb',as_version=4)
nbformat.validate(b)
codes=[c.source for c in b.cells if c.cell_type=='code']
for s in codes:compile(s,'lr_t1_poly2.ipynb','exec')
known=set(dir(builtins))|{'__name__'}
def loads(n):return {x.id for x in ast.walk(n) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load)}
def check(n):
 missing=loads(n)-known
 assert not missing,missing
def targets(n):return {x.id for x in ast.walk(n) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Store)}
def statements(nodes):
 for n in nodes:
  if isinstance(n,(ast.Import,ast.ImportFrom)):
   known.update(x.asname or x.name.split('.')[0] for x in n.names)
  elif isinstance(n,ast.FunctionDef):
   for x in n.args.defaults:check(x)
   table=symtable.symtable(ast.unparse(n),'<build>','exec').get_children()[0]
   assert not {x.get_name() for x in table.get_symbols() if x.is_global() and x.is_referenced()}-known
   known.add(n.name)
  elif isinstance(n,ast.Assign):
   check(n.value)
   for x in n.targets:known.update(targets(x))
  elif isinstance(n,ast.For):
   check(n.iter);known.update(targets(n.target));statements(n.body);statements(n.orelse)
  elif isinstance(n,ast.If):check(n.test);statements(n.body);statements(n.orelse)
  elif isinstance(n,ast.Expr):
   # The term-count table has one comprehension-local target.
   comprehensions=[x for x in ast.walk(n) if isinstance(x,ast.comprehension)]
   local=set().union(*(targets(x.target) for x in comprehensions)) if comprehensions else set()
   assert not loads(n)-(known|local),loads(n)-(known|local)
  elif isinstance(n,ast.Raise):check(n.exc)
  elif isinstance(n,ast.Break):pass
  else:raise AssertionError(type(n).__name__)
statements(ast.parse('\n\n'.join(codes)).body)
imports={x.asname or x.name.split('.')[0] for n in ast.walk(ast.parse('\n\n'.join(codes))) if isinstance(n,(ast.Import,ast.ImportFrom)) for x in n.names}
assert imports<=loads(ast.parse('\n\n'.join(codes)))
def builder(nb):
 fn=next(n for c in nb.cells if c.cell_type=='code' for n in ast.parse(c.source).body if isinstance(n,ast.FunctionDef) and n.name=='build')
 env={'make_pipeline':make_pipeline,'build_preprocessor':build_preprocessor,'PolynomialFeatures':PolynomialFeatures,
      'StandardScaler':StandardScaler,'LogisticRegression':LogisticRegression,'RANDOM_STATE':RANDOM_STATE}
 exec(compile(ast.Module(body=[fn],type_ignores=[]),'<build-only>','exec'),env)
 return env['build']
old,new=builder(a),builder(b)
for C in [.001,.01,.1,1,10,100]:
 for budget in [2000,4000,8000]:
  p,q=old(C,budget),new(C,budget)
  assert p.steps[-1][1].get_params()==q.steps[-1][1].get_params()
  assert p.named_steps['polynomialfeatures'].get_params()==q.named_steps['polynomialfeatures'].get_params()
  assert p.named_steps['standardscaler'].get_params()==q.named_steps['standardscaler'].get_params()
  assert repr(p.steps[0][1])==repr(q.steps[0][1])
print('Notebook structure, syntax, definition order, imports and unfitted parameters pass. No training performed by this check.')
