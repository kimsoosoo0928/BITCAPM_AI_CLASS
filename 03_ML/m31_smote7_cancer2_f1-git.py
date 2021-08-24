from numpy.lib.function_base import average
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import time
import warnings
warnings=warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, f1_score

datasets = load_breast_cancer()
x = datasets['data']
y = datasets['target']

