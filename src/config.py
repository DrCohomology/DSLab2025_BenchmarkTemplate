"""
This file
    - defines macros
    - defines datasets and their openml ids
    - imports and initializes the objects for main.py
"""

# General parameters
RANDOM_STATE = 1
MAIN_PARAMETERS = {
    "n_splits": 5,  # splits of cross-validation
    "timeout": 6000,  # maximum allowed run time in seconds
}
NUM_PROCESSES = 1  # pass to joblib.Parallel for parallel execution


# Encoders
rlibs = None
std = [e.BinaryEncoder(), e.CatBoostEncoder(), e.CountEncoder(), e.DropEncoder(), e.MinHashEncoder(), e.OneHotEncoder(),
       e.OrdinalEncoder(), e.RGLMMEncoder(rlibs=rlibs), e.SumEncoder(), e.TargetEncoder(), e.WOEEncoder()]
cvglmm = [e.CVRegularized(e.RGLMMEncoder(rlibs=rlibs), n_splits=ns) for ns in [2, 5, 10]]
cvte = [e.CVRegularized(e.TargetEncoder(), n_splits=ns) for ns in [2, 5, 10]]
buglmm = [e.CVBlowUp(e.RGLMMEncoder(rlibs=rlibs), n_splits=ns) for ns in [2, 5, 10]]
bute = [e.CVBlowUp(e.TargetEncoder(), n_splits=ns) for ns in [2, 5, 10]]
dte = [e.Discretized(e.TargetEncoder(), how="minmaxbins", n_bins=nb) for nb in [2, 5, 10]]
binte = [e.PreBinned(e.TargetEncoder(), thr=thr) for thr in [1e-3, 1e-2, 1e-1]]
me = [e.MeanEstimateEncoder(m=m) for m in [1e-1, 1, 10]]
ENCODERS = reduce(lambda x, y: x+y, [std, cvglmm, cvte, buglmm, bute, dte, binte, me])

# Datasets with their OpenML id (these are for binary classification)
DATASETS = MappingProxyType({
    'kr-vs-kp': 3,  # https://archive.ics.uci.edu/dataset/22/chess+king+rook+vs+king+pawn
    'credit-approval': 29,  # http://archive.ics.uci.edu/dataset/27/credit+approval
    'credit-g': 31,  # https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
    'sick': 38,  # http://archive.ics.uci.edu/dataset/102/thyroid+disease
    'tic-tac-toe': 50,  # http://archive.ics.uci.edu/dataset/101/tic+tac+toe+endgame
    'heart-h': 51,  # https://archive.ics.uci.edu/dataset/45/heart+disease
    'vote': 56,  # https://archive.ics.uci.edu/dataset/105/congressional+voting+records
    'monks-problems-1': 333,  # https://archive.ics.uci.edu/dataset/70/monk+s+problems
    'monks-problems-2': 334,  # https://archive.ics.uci.edu/dataset/70/monk+s+problems
    'irish': 451,  # http://lib.stat.cmu.edu/datasets/irish.ed
    'profb': 470,  # http://lib.stat.cmu.edu/datasets/profb
    'mv': 881,  # https://www.openml.org/search?type=data&status=active&id=881
    'molecular-biology_promoters': 956,
    # https://archive.ics.uci.edu/dataset/67/molecular+biology+promoter+gene+sequences
    'nursery': 959,  # https://www.openml.org/search?type=data&status=active&id=26
    'kdd_internet_usage': 981,  # https://www.openml.org/search?type=data&status=active&id=4133
    'ada_prior': 1037,  # https://www.agnostic.inf.ethz.ch/datasets.php
    'KDDCup09_appetency': 1111,  # https://www.openml.org/search?type=data&status=active&id=1111&sort=runs
    'KDDCup09_churn': 1112,  # https://www.openml.org/search?type=data&status=active&id=1112&sort=runs
    'KDDCup09_upselling': 1114,  # https://www.openml.org/search?type=data&status=active&id=1114
    'airlines': 1169,  # https://www.openml.org/search?type=data&status=active&id=1169
    'Agrawal1': 1235,  # https://www.openml.org/search?type=data&status=active&id=1235
    'bank-marketing': 1461,  # https://archive.ics.uci.edu/dataset/222/bank+marketing
    'blogger': 1463,  # https://www.ijcaonline.org/archives/volume47/number18/7291-0509
    'nomao': 1486,  # https://archive.ics.uci.edu/dataset/227/nomao
    'thoracic-surgery': 1506,  # https://www.openml.org/search?type=data&status=active&id=1506
    'wholesale-customers': 1511,  # https://www.openml.org/search?type=data&status=active&id=1511
    'adult': 1590,  # https://www.openml.org/search?type=data&status=active&id=1590
    'amazon_employee_access': 43900,
    # 4135                               # https://www.kaggle.com/competitions/amazon-employee-access-challenge/data
    'cylinder-bands': 6332,  # https://archive.ics.uci.edu/dataset/32/cylinder+bands
    'dresses-sales': 23381,  # https://archive.ics.uci.edu/dataset/289/dresses+attribute+sales
    'SpeedDating': 40536,  # https://www.openml.org/search?type=data&status=active&id=40536
    'Titanic': 40945,  # https://www.openml.org/search?type=data&status=active&id=40945
    'Australian': 40981,  # https://archive.ics.uci.edu/dataset/143/statlog+australian+credit+approval
    'jungle_chess_2pcs_endgame_elephant_elephant': 40999,
    # https://www.openml.org/search?type=data&status=active&id=40999
    'jungle_chess_2pcs_endgame_rat_rat': 41005,  # https://www.openml.org/search?type=data&status=active&id=41005
    'jungle_chess_2pcs_endgame_lion_lion': 41007,  # https://www.openml.org/search?type=data&status=active&id=41007
    'kick': 41162,  # https://www.openml.org/search?type=data&status=active&id=41162
    'porto-seguro': 41224,  # https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction
    'telco-customer-churn': 42178,  # https://www.kaggle.com/datasets/blastchar/telco-customer-churn/discussion
    'KDD98': 42343,  # https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html
    'sf-police-incidents': 42344,  # https://www.openml.org/search?type=data&status=active&id=42344
    'open_payments': 42738,  # https://www.openml.org/search?type=data&status=active&id=42738
    'Census-Income-KDD': 42750,  # https://www.openml.org/search?type=data&status=active&id=42750
    'students_scores': 43098,  # https://www.openml.org/search?type=data&status=active&id=43098
    'WMO-Hurricane-Survival-Dataset': 43607,  # https://www.openml.org/search?type=data&status=active&id=43607
    'law-school-admission-bianry': 43890,  # https://www.openml.org/search?type=data&status=active&id=43890
    'national-longitudinal-survey-binary': 43892,  # https://www.openml.org/search?type=data&status=active&id=43892
    'ibm-employee-attrition': 43896,  # https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
    'ibm-employee-performance': 43897,
    # https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
    'mushroom': 43922  # https://www.openml.org/search?type=data&status=active&id=24
})
DATASETS_NAMES, DATASETS_IDS = list(DATASETS.keys()), list(DATASETS.values())

# Some random models
MODELS = {
    DecisionTreeClassifier(random_state=RANDOM_STATE + 2, max_depth=5),
    SVC(random_state=RANDOM_STATE + 4, C=1.0, kernel="rbf", gamma="scale", probability=True),
    KNeighborsClassifier(n_neighbors=5),
    LogisticRegression(max_iter=100, random_state=RANDOM_STATE + 6, solver="lbfgs"),
    DecisionTreeClassifier(random_state=RANDOM_STATE+2),
    SVC(random_state=RANDOM_STATE+4, probability=True),
    KNeighborsClassifier(),
    LogisticRegression(max_iter=100, random_state=RANDOM_STATE+6, solver="lbfgs"),
    LGBMClassifier(random_state=RANDOM_STATE+3, n_estimators=3000, metric="None"),  # LGBM needs early_stopping
}

# Quality metrics (for binary classification)
SCORINGS = [accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score]

# Other pre-processing classes
SCALERS = [RobustScaler()]
IMPUTERS_CAT = [e.DFImputer(SimpleImputer(strategy="most_frequent"))]
IMPUTERS_NUM = [e.DFImputer(SimpleImputer(strategy="median"))]
