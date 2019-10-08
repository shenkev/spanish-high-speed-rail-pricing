import numpy as np
import xgboost as xgb
import random
from tqdm import tqdm

DATA_FILE = "./data/ready_data_0.01.npz"

loaded = np.load(DATA_FILE)
[train_X, train_y, val_X, val_y] = [loaded[x] for x in ["train_X", "train_y", "val_X", "val_y"]]


dtrain = xgb.DMatrix(train_X, label=train_y)
dval = xgb.DMatrix(val_X, label=val_y)

all_val_rmse = []
all_val_rmse_map = {}

def hyper_search(experiments=10000):

    for _ in tqdm(range(experiments)):
        one_experiment()

def one_experiment():

    # params
    num_round = random.randint(2,200)

    param = {
    'objective':'reg:squarederror',
    'eval_metric':"rmse",
    'verbosity':0, 
    # 'booster': 'dart',
    'eta': 0.1*(3.0**random.uniform(-4.0, 4.0)),
    # 'gamma': 0,
    'max_depth':random.randint(2,50),
    'subsample': random.uniform(0.25, 1.0),
    'colsample_bytree': random.uniform(0.25, 1.0),
    # 'colsample_bynode': random.uniform(0.25, 1.0),
    'lambda': 1.0*(2.0**random.uniform(-4.0, 4.0)),
    'alpha': 0.0001*(10.0**random.uniform(0.0, 5.0)),

    # 'rate_drop': 0.0,
    # 'skip_drop': 0.0
    }

    # train
    watchlist = [(dval, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=10, verbose_eval=False)

    # prediction
    preds = bst.predict(dval, ntree_limit=bst.best_ntree_limit)
    labels = dval.get_label()

    # best score
    val_perf = np.sqrt(np.mean((preds-labels)**2))
    print('BEST VAL RMSE: {}'.format(val_perf))

    # save if best model
    if len(all_val_rmse) == 0 or val_perf < min(all_val_rmse):
        bst.save_model('./model/best.model')

    # save results
    all_val_rmse.append(val_perf)
    all_val_rmse_map[val_perf] = [bst.best_score, param]


hyper_search()

print("done")