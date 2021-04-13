#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 21:52:13 2020

@author: SYang
"""

import seaborn as sns

filecorrA = '/home/kalman/temp/feature_dose_2hr_A.xlsx'
filecorrB = '/home/kalman/temp/feature_dose_2hr_B.xlsx'
datamiA1 = pd.read_excel(filecorrA, 3)
datamiB1 = pd.read_excel(filecorrB, 3)
datamiA2 = pd.read_excel(filecorrA, 4)
datamiB2 = pd.read_excel(filecorrB, 4)


datamiA1['group'] = ['A'] * 474
datamiB1['group'] = ['B'] * 475
datamiA2['group'] = ['A'] * 474
datammi
sns.set(style="whitegrid", palette="pastel", color_codes=True)


# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(x="day", y="total_bill", hue="smoker",
               split=True, inner="quart",
               palette={"Yes": "y", "No": "b"},
               data=tips)
sns.despine(left=True)