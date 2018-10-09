# Copyright 2018 Cognibit Solutions LLP.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""

Generates classification report for the trained XGBoost models
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report as report

def preprocessing(results, truth):
	# preprocessing
	results.loc[truth['before']==truth['after'],'truth']='RemainSelf'
	results.loc[truth['before']!=truth['after'],'truth']='ToBeNormalized'
	truth['class']=''
	truth.loc[truth['before']!=truth['after'],'class']='ToBeNormalized'
	truth.loc[truth['before']==truth['after'],'class']='RemainSelf'
	return results, truth

def f1_scores(results, truth):
	print(report(truth['class'].tolist(), results['class'].tolist()))

def confusion_matrix(results, truth, lang):
	matrix = cm(truth['class'].tolist(), results['class'].tolist())
	plot_confusion_matrix(matrix, classes=['ToBeNormalized', 'RemainSelf'], 
		title='XGBoost Confusion Matrix [{}]'.format(lang))
	
def pr_curve(results, truth, lang):
	truth.loc[truth['class']=='ToBeNormalized', 'class'] = 1
	truth.loc[truth['class']=='RemainSelf', 'class'] = 0
	results.loc[results['class']=='ToBeNormalized', 'class'] = 1
	results.loc[results['class']=='RemainSelf', 'class'] = 0

	average_precision = average_precision_score(truth['class'].tolist(), results['class'].tolist()) 
	precision, recall, threshold = precision_recall_curve(truth['class'].tolist(), results['class'].tolist())

	plt.step(recall, precision, color='b', alpha=0.2, where='post')
	plt.fill_between(recall, precision, alpha=0.2, color='b')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('Precision-Recall Curve: AP={0:0.2f} [{1}]'.format(average_precision, lang))
	plt.show()

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

