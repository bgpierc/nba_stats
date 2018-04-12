import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, SoupStrainer
import sys
import string
import requests
import datetime
import progressbar
import time
import re
import matplotlib.pyplot as plt
import matplotlib
import tkinter
import statsmodels.api as sm


def NCAAft_toNBAfgpct():
	df = pd.read_csv('players.csv')
	X = []
	y = []
	for i in range(0,len(df)):
		if df['NBA_fga_per_game'][i] > 3 and df['NCAA_fgapg'][i] > 3:
			if df['NCAA_ft'][i] >0 and df['NBA_fg%'][i] > 0:
				if df['NCAA_ft'][i] <1 and df['NBA_fg%'][i] < 1:
					X.append(df['NCAA_ft'][i])
					y.append(df['NBA_fg%'][i])
	model = sm.OLS(y, X).fit()
	print (model.summary())
	plt.plot(X, model.fittedvalues)
	plt.scatter(X,y, c = 'black', marker = '.')
	plt.xlabel('NCAA ft%')
	plt.ylabel('NBA fg%')
	plt.show()

NCAAft_toNBAfgpct()