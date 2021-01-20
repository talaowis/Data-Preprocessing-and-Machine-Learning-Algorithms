import pandas as pd
import pandasql as ps
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
#imbalanced-learn version==0.5
from imblearn.over_sampling import SMOTENC
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numpy import where
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
#Classification Algorithms
from sklearn.tree import DecisionTreeClassifier,plot_tree # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
#visualizing Decision Trees
from sklearn.tree import export_graphviz
import graphviz
import pickle
#feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from random import randrange
#SVm 
from sklearn import svm
from sklearn.model_selection import learning_curve, GridSearchCV
from mpl_toolkits import mplot3d
from ipywidgets import interact, fixed
#KNN
from sklearn.neighbors import KNeighborsClassifier
#Neural Network
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

number_of_dimensions=9
tree_max_depth = 4