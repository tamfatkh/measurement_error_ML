#!/usr/bin/env python3
"""
run_all_be.py: Generates Berkson measurement-error datasets and runs both regression and classification
Optuna-based pipelines on feature sets X, W1, W2 using multiple models. Saves results incrementally.
"""
# to test the code use: python3 -m py_compile run_all_be.py
# to run the code use: python3 run_all_be.py
# pip install --upgrade pip
# pip install numpy pandas optuna xgboost scikit-learn


import os
import json
import random
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import multiprocessing
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.inspection import permutation_importance
import warnings
from sklearn.exceptions import ConvergenceWarning
import inspect

# suppress warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# === 1) DATA GENERATION ===
def simulate_berkson_data(n, a0, a1, a2, a3):
    W = np.random.normal(0, 1, size=n)
    X1 = W + np.random.normal(0, .5, size=n)
    X2 = W + np.random.normal(0, .75, size=n)
    Z1 = np.random.normal(0, 1, size=n)
    Z2 = np.random.normal(0, 1, size=n)

    L1 = a0 + a1 * X1 + a2 * Z1 + a3 * Z2
    Y1 = L1 + np.random.normal(0, 0.1, size=n)
    p1 = 1 / (1 + np.exp(-L1))
    B1 = np.random.binomial(1, p1, size=n)

    L2 = a0 + a1 * X2 + a2 * Z1 + a3 * Z2
    Y2 = L2 + np.random.normal(0, 0.1, size=n)
    p2 = 1 / (1 + np.exp(-L2))
    B2 = np.random.binomial(1, p2, size=n)

    return X1, X2, W, Z1, Z2, Y1, Y2, p1, B1, p2, B2


def generate_all_data(param_sets, n_simulations=10, outdir="BEdata"):
    os.makedirs(outdir, exist_ok=True)
    for idx, (n, a0, a1, a2, a3) in enumerate(param_sets, start=1):
        n = int(n)
        for i in range(1, n_simulations+1):
            X1, X2, W, Z1, Z2, Y1, Y2, p1, B1, p2, B2 = simulate_berkson_data(n, a0, a1, a2, a3)
            df = pd.DataFrame({"X1":X1, "X2":X2, "W":W, "Z1":Z1, "Z2":Z2, "Y1":Y1, "Y2":Y2, "p1":p1, "B1":B1, "p2":p2, "B2":B2})
            train_idx1, test_idx1 = train_test_split(df.index, test_size=0.2, stratify=df['B1'], random_state=42)
            df['train_test_split1'] = 0
            df.loc[train_idx1, 'train_test_split1'] = 1
            train_idx2, test_idx2 = train_test_split(df.index, test_size=0.2, stratify=df['B2'], random_state=42)
            df['train_test_split2'] = 0
            df.loc[train_idx2, 'train_test_split2'] = 1
            df.to_csv(f"{outdir}/BE_data_with_split_{idx}_{i}.csv", index=False)

# === 2) SEEDING ===
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# === 3) EVALUATION HELPERS ===
def train_eval_reg(model_name, best_params, X_tr, X_te, y_tr, y_te):
    # 1) Copy so we don’t clobber Optuna’s best_params
    params = best_params.copy()

    # 2) Inject random_state where supported
    if model_name not in ("SVR", "MLR"):
        params["random_state"] = 42

    # 3) Special‑case MLPRegressor to build hidden_layer_sizes
    if model_name == "MLP-R":
        # pop out the layer count
        n_layers = params.pop("n_layers")
        # collect exactly n_layers hidden_size_i entries in order
        hidden_sizes = tuple(params.pop(f"hidden_size_{i}") for i in range(n_layers))
        params["hidden_layer_sizes"] = hidden_sizes

    # 4) Filter params to only those the constructor cares about
    constructor = {
      "MLR": LinearRegression,
      "SVR": SVR,
      "RF-R": RandomForestRegressor,
      "XGBoost-R": lambda **kw: xgb.XGBRegressor(**kw, eval_metric='rmse'),
      "MLP-R": MLPRegressor
    }[model_name]

    # Optionally introspect to drop any unknown keys:
    # import inspect
    sig = inspect.signature(constructor.__init__)
    valid_params = {k: v for k, v in params.items() if k in sig.parameters}

    model = constructor(**valid_params)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    r2 = r2_score(y_te, preds)
    perm = permutation_importance(
        model, X_te, y_te,
        n_repeats=10, random_state=42
    ).importances_mean
    return r2, dict(zip(X_te.columns, perm))

def train_eval_clf(model_name, best_params, X_tr, X_te, y_tr, y_te):
    # 1) Copy so we don’t clobber Optuna’s best_params
    params = best_params.copy()

    # 2) Inject random_state where supported
    if model_name not in ("SVC", "LogReg"):
        params["random_state"] = 42

    # 3) Special‑case MLPClassifier to build hidden_layer_sizes
    if model_name == "MLP-C":
        # pop out the layer count
        n_layers = params.pop("n_layers")
        # collect exactly n_layers hidden_size_i entries in order
        hidden_sizes = tuple(params.pop(f"hidden_size_{i}") for i in range(n_layers))
        params["hidden_layer_sizes"] = hidden_sizes

    # 4) Constructor lookup
    constructors = {
        "LogReg": LogisticRegression,
        "SVC": lambda **kw: SVC(**kw, probability=True),
        "RF-C": RandomForestClassifier,
        "XGBoost-C": lambda **kw: xgb.XGBClassifier(**kw, eval_metric='logloss'),
        "MLP-C": MLPClassifier
    }
    constructor = constructors[model_name]

    # 5) Drop any params not in the constructor signature
    sig = inspect.signature(constructor.__init__)
    valid_params = {k: v for k, v in params.items() if k in sig.parameters}

    # 6) Instantiate, fit, and evaluate
    model = constructor(**valid_params)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    acc = accuracy_score(y_te, preds)
    f1  = f1_score(y_te, preds)
    perm = permutation_importance(
        model, X_te, y_te,
        n_repeats=10, random_state=42
    ).importances_mean

    return acc, f1, dict(zip(X_te.columns, perm))


# === 4) OBJECTIVES DEFINITIONS ===
def define_objectives():
    objs = {}
    # Regression
    objs["MLR"] = lambda t, Xtr, Xte, ytr, yte: mean_squared_error(
        yte, LinearRegression(
            fit_intercept=t.suggest_categorical("fit_intercept", [True, False]),
            positive=t.suggest_categorical("positive", [True, False])
        ).fit(Xtr, ytr).predict(Xte)
    )
    objs["SVR"] = lambda t, Xtr, Xte, ytr, yte: mean_squared_error(
        yte, SVR(
            C=t.suggest_float("C", 0.1, 10.0),
            epsilon=t.suggest_float("epsilon", 0.01, 1.0),
            kernel=t.suggest_categorical("kernel", ["linear","poly","rbf","sigmoid"])
        ).fit(Xtr, ytr).predict(Xte)
    )
    objs["RF-R"] = lambda t, Xtr, Xte, ytr, yte: mean_squared_error(
        yte, RandomForestRegressor(
            n_estimators=t.suggest_int("n_estimators",10,200),
            max_depth=t.suggest_int("max_depth",2,20),
            min_samples_split=t.suggest_int("min_samples_split",2,5),
            min_samples_leaf=t.suggest_int("min_samples_leaf",1,2),
            max_features=t.suggest_categorical("max_features",["sqrt","log2"]),
            random_state=42
        ).fit(Xtr,ytr).predict(Xte)
    )
    objs["XGBoost-R"] = lambda t, Xtr, Xte, ytr, yte: mean_squared_error(
        yte, xgb.XGBRegressor(
            n_estimators=t.suggest_int("n_estimators",100,200),
            learning_rate=t.suggest_float("learning_rate",0.01,0.1),
            max_depth=t.suggest_int("max_depth",3,5),
            subsample=t.suggest_float("subsample",0.8,1.0),
            colsample_bytree=t.suggest_float("colsample_bytree",0.5,1.0),
            random_state=42
        ).fit(Xtr,ytr).predict(Xte)
    )
    objs["MLP-R"] = lambda t, Xtr, Xte, ytr, yte: mean_squared_error(
        yte, MLPRegressor(
            hidden_layer_sizes=tuple(
                t.suggest_int(f"hidden_size_{i}",10,100) for i in range(t.suggest_int("n_layers",1,3))
            ),
            activation=t.suggest_categorical("activation",["identity","logistic","tanh","relu"]),
            solver=t.suggest_categorical("solver",["lbfgs","sgd","adam"]),
            learning_rate_init=t.suggest_float("learning_rate_init",0.001,0.01, log=True),
            alpha=t.suggest_float("alpha",0.0001,0.1,log=True),
            max_iter=t.suggest_int("max_iter",500,2000),
            random_state=42
        ).fit(Xtr,ytr).predict(Xte)
    )
    # Classification
    objs["LogReg"] = lambda t, Xtr, Xte, ytr, yte: 1 - accuracy_score(
        yte, LogisticRegression(
            penalty=t.suggest_categorical("penalty",["l2"]),
            solver=t.suggest_categorical("solver", ["lbfgs","liblinear","saga"]),
            C=t.suggest_float("C",0.01,10.0,log=True),
            max_iter=t.suggest_int("max_iter",100,500),
            random_state=42
        ).fit(Xtr,ytr).predict(Xte)
    )
    objs["SVC"] = lambda t, Xtr, Xte, ytr, yte: 1 - accuracy_score(
        yte, SVC(
            C=t.suggest_float("C",0.1,10.0),
            kernel=t.suggest_categorical("kernel",["linear","poly","rbf","sigmoid"]),
            probability=True, random_state=42
        ).fit(Xtr,ytr).predict(Xte)
    )
    objs["RF-C"] = lambda t, Xtr, Xte, ytr, yte: 1 - accuracy_score(
        yte, RandomForestClassifier(
            n_estimators=t.suggest_int("n_estimators",10,200),
            max_depth=t.suggest_int("max_depth",2,20),
            min_samples_split=t.suggest_int("min_samples_split",2,5),
            min_samples_leaf=t.suggest_int("min_samples_leaf",1,2),
            max_features=t.suggest_categorical("max_features",["sqrt","log2"]),
            random_state=42
        ).fit(Xtr,ytr).predict(Xte)
    )
    objs["XGBoost-C"] = lambda t, Xtr, Xte, ytr, yte: 1 - accuracy_score(
        yte, xgb.XGBClassifier(
            n_estimators=t.suggest_int("n_estimators",100,200),
            learning_rate=t.suggest_float("learning_rate",0.01,0.1),
            max_depth=t.suggest_int("max_depth",3,5),
            subsample=t.suggest_float("subsample",0.8,1.0),
            colsample_bytree=t.suggest_float("colsample_bytree",0.5,1.0),
            eval_metric="logloss", random_state=42
        ).fit(Xtr,ytr).predict(Xte)
    )
    objs["MLP-C"] = lambda t, Xtr, Xte, ytr, yte: 1 - accuracy_score(
        yte, MLPClassifier(
            hidden_layer_sizes=tuple(
                t.suggest_int(f"hidden_size_{i}",10,100) for i in range(t.suggest_int("n_layers",1,3))
            ),
            activation=t.suggest_categorical("activation",["identity","logistic","tanh","relu"]),
            solver=t.suggest_categorical("solver",["lbfgs","sgd","adam"]),
            alpha=t.suggest_float("alpha",0.0001,0.1,log=True),
            learning_rate_init=t.suggest_float("learning_rate_init",0.001,0.01, log=True),
            max_iter=t.suggest_int("max_iter",500,2000),
            random_state=42
        ).fit(Xtr,ytr).predict(Xte)
    )
    return objs


    
# === 5) TASK RUNNER (file‑based) ===
def run_optimization(task):
    """
    task = (filepath, idx, sim)
    Each worker reads exactly one CSV, processes it, then discards it.
    """
    filepath, idx, sim = task

    # 1) Load & split
    df = pd.read_csv(filepath)
    scaler = StandardScaler() # MinMaxScaler()
    df[['X1','X2', 'W', 'Z1','Z2','Y1','Y2']] = scaler.fit_transform(
        df[['X1','X2', 'W','Z1','Z2','Y1','Y2']])
    train1 = df[df.train_test_split1 == 1]
    test1  = df[df.train_test_split1 == 0]
    train2 = df[df.train_test_split2 == 1]
    test2  = df[df.train_test_split2 == 0]

    # 2) Pull out features & targets
    Xtr_W1, Xte_W1 = train1[['W','Z1','Z2']], test1[['W','Z1','Z2']]
    Xtr_W2, Xte_W2 = train2[['W','Z1','Z2']], test2[['W','Z1','Z2']]
    ytr_reg1, yte_reg1 = train1.Y1, test1.Y1
    ytr_reg2, yte_reg2 = train2.Y2, test2.Y2

    Xtr_X1, Xte_X1 = train1[['X1','Z1','Z2']], test1[['X1','Z1','Z2']]
    Xtr_X2, Xte_X2 = train2[['X2','Z1','Z2']], test2[['X2','Z1','Z2']]
    ytr_clf1, yte_clf1 = train1.B1, test1.B1
    ytr_clf2, yte_clf2 = train2.B2, test2.B2


    # 3) Run your Optuna + eval, exactly as before
    results = []
    objs = define_objectives()
    reg_models = ["MLR","SVR","RF-R","XGBoost-R","MLP-R"]
    clf_models = ["LogReg","SVC","RF-C","XGBoost-C","MLP-C"]
    
    def safe_obj(model_key, trial, Xtr, Xte, ytr, yte):
        try:
            return objs[model_key](trial, Xtr, Xte, ytr, yte)
        except ValueError:
            # any divergence/non‑finite weights → worst possible score
            return float("inf")
    
    for m in reg_models:
        for label, Xtr, Xte in [("W1",Xtr_W1,Xte_W1),("W2",Xtr_W2,Xte_W2),("X1",Xtr_X1,Xte_X1),("X2",Xtr_X2,Xte_X2)]:
            study = optuna.create_study(direction="minimize",
                                        pruner=optuna.pruners.MedianPruner())
            #study.optimize(lambda t: objs[m](t,Xtr,Xte,ytr_reg,yte_reg),
            if label in ("W1", "X1"):
                study.optimize(lambda t: safe_obj(m, t, Xtr, Xte, ytr_reg1, yte_reg1),
                           n_trials=20, catch=(ValueError,), show_progress_bar=False)
            
                r2, perm = train_eval_reg(m, study.best_params,
                                     Xtr, Xte, ytr_reg1, yte_reg1)
            else:
                study.optimize(lambda t: safe_obj(m, t, Xtr, Xte, ytr_reg2, yte_reg2),
                           n_trials=20, catch=(ValueError,), show_progress_bar=False)
                r2, perm = train_eval_reg(m, study.best_params,
                                     Xtr, Xte, ytr_reg2, yte_reg2)
            results.append([idx, sim, label, m,
                            study.best_value, r2, None,
                            json.dumps(study.best_params),
                            json.dumps(perm)])

    for m in clf_models:
        for label, Xtr, Xte in [("W1",Xtr_W1,Xte_W1),("W2",Xtr_W2,Xte_W2),("X1",Xtr_X1,Xte_X1),("X2",Xtr_X2,Xte_X2)]:
            study = optuna.create_study(direction="minimize",
                                        pruner=optuna.pruners.MedianPruner())
            #study.optimize(lambda t: objs[m](t,Xtr,Xte,ytr_clf,yte_clf),
            if label in ("W1", "X1"):
                study.optimize(lambda t: safe_obj(m, t, Xtr, Xte, ytr_clf1, yte_clf1),
                           n_trials=20, catch=(ValueError,), show_progress_bar=False)
                acc, f1, perm = train_eval_clf(m, study.best_params,
                                          Xtr, Xte, ytr_clf1, yte_clf1)
            else:
                study.optimize(lambda t: safe_obj(m, t, Xtr, Xte, ytr_clf2, yte_clf2),
                           n_trials=20, catch=(ValueError,), show_progress_bar=False)
                acc, f1, perm = train_eval_clf(m, study.best_params,
                                          Xtr, Xte, ytr_clf2, yte_clf2)
            results.append([idx, sim, label, m,
                            study.best_value, acc, f1,
                            json.dumps(study.best_params),
                            json.dumps(perm)])

    out_fn = f"BEdata/results_{idx}_{sim}.csv"
    pd.DataFrame(results, columns=[
        "Index","Simulation","Feature","Model","Loss_Error",
        "Metric1","Metric2","BestParams","PermImportance"
    ]).to_csv(out_fn, index=False)
    return f"{out_fn} saved"


# === 6) TASK COLLECTION ===
def collect_tasks(datadir="BEdata"):
    tasks=[]
    for fn in sorted(os.listdir(datadir)):
        if not fn.endswith(".csv"): continue
        parts = fn.rstrip(".csv").split("_")
        idx, sim = int(parts[-2]), int(parts[-1])
        tasks.append((os.path.join(datadir, fn), idx, sim))
    return tasks

# === 7) MAIN ===
if __name__=="__main__":
    set_seeds(42)
    
    param_sets=[(2000,0,2,1,0),(2000,0,1,2,0),(2000,0,2,2,0),(2000,1,2,1,0),(2000,1,1,2,0),(2000,1,2,2,0),
                (3000,0,2,1,0),(3000,0,1,2,0),(3000,0,2,2,0),(3000,1,2,1,0),(3000,1,1,2,0),(3000,1,2,2,0),
                (4000,0,2,1,0),(4000,0,1,2,0),(4000,0,2,2,0),(4000,1,2,1,0),(4000,1,1,2,0),(4000,1,2,2,0)]
    
    # param_sets=[(2000,0,2,1,0)]
    n_sim = 10
    if not os.path.isdir("BEdata") or len([f for f in os.listdir("BEdata") if f.startswith("BE_data_with_split_")])<len(param_sets)*n_sim:
        generate_all_data(param_sets,n_sim,"BEdata")
    tasks=collect_tasks("BEdata")
    # max_avail = multiprocessing.cpu_count()
    # n_cpus = min(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), max_avail)
    n_cpus = 56
    with Pool(n_cpus) as pool:
        pool.map(run_optimization,tasks)
    # merge
    import glob
    files=glob.glob("BEdata/results_*.csv")
    pd.concat([pd.read_csv(f) for f in files],ignore_index=True).to_csv("BEdata/combined_results_R_and_C.csv",index=False)
