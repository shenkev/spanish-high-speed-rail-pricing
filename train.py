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
    num_round = random.randint(80, 80)

    param = {
    'objective':'reg:squarederror',
    'eval_metric':"rmse",
    'verbosity':0, 
    # 'booster': 'dart',
    'eta': 0.1*(3.0**random.uniform(0.0, 0.0)),
    # 'gamma': 0,
    'max_depth':random.randint(25, 25),
    'subsample': random.uniform(0.8, 0.8),
    # 'colsample_bytree': random.uniform(1.0, 1.0),
    'colsample_bynode': random.uniform(0.65, 0.65),
    'lambda': 20.0*(5.0**random.uniform(0.0, 0.0)),
    'alpha': 5.0*(5.0**random.uniform(0.0, 0.0)),

    # 'rate_drop': random.uniform(0.015, 0.05),
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

    # sorted_val = sorted(all_val_rmse)
    # all_val_rmse_map[sorted_val[-1]]
    # alpha = [all_val_rmse_map[sorted_val[i]][1]['alpha'] for i in range(len(sorted_val))]

    # save results
    param['rounds'] = num_round
    all_val_rmse.append(val_perf)
    all_val_rmse_map[val_perf] = [bst.best_score, param]
    print(val_perf)
    print(bst.best_score)


hyper_search()

print("done")