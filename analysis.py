import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, SoupStrainer
import sys
import string
import requests
import datetime
import re
import matplotlib.pyplot as plt
import matplotlib
import tkinter
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.utils import shuffle
from scipy import stats
import seaborn as sns
import scipy.stats as st
import sklearn.linear_model as lm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.formula.api as smf

def predictFromLine(slope, intercept,x):
	return slope*x + intercept
def NBAft():
	df = pd.read_csv('players.csv')
	df = shuffle(df)
	Xtrain = []
	ytrain = []
	Xtest = []
	ytest = []
	for i in range(0,int(len(df)/2)):
		if df['NBA_ft_per_g'][i] > 0 and df['NCAA_ftapg'][i] > 0:
			if df['NCAA_ft'][i] >0 and df['NBA_ft%'][i] > 0:
				if df['NCAA_ft'][i] <1 and df['NBA_ft%'][i] < 1:
					Xtrain.append(df['NCAA_ft'][i])
					ytrain.append(df['NBA_ft%'][i])
	for i in range(int(len(df)/2),len(df)):
		if df['NBA_ft_per_g'][i] > 0 and df['NCAA_ft'][i] > 0:
			if df['NCAA_ft'][i] >0 and df['NBA_ft%'][i] > 0:
				if df['NCAA_ft'][i] <1 and df['NBA_ft%'][i] < 1:
					Xtest.append(df['NCAA_ft'][i])
					ytest.append(df['NBA_fg%'][i])
	x = np.linspace(0.2,1,100)

	slope, intercept, r_value, p_value, std_err = stats.linregress(Xtrain,ytrain)
	plt.figure(1)
	plt.scatter(Xtrain,ytrain,c = 'black', marker = '.')
	plt.plot(x,x*slope + intercept)
	plt.plot()
	plt.show()

def efgpct():
	df = pd.read_csv('players.csv')
	df = df[['NBA_efgpct','NCAA_ft']]
	X = df['NCAA_ft'].as_matrix()
	y = df['NBA_efgpct'].as_matrix()
	Xf = []
	yf = []
	for i in range(0,len(X)):
		if X[i] > 0 and X[i] < 1 and y[i] > 0 and y[i] < 1:
			Xf.append(X[i])
			yf.append(y[i])
	plt.figure(1)
	plt.scatter(Xf,yf, color = 'black', marker = '.')
	slope, intercept, r_value, p_value, std_err = stats.linregress(Xf,yf)
	x = np.linspace(0.3,0.9, 1000)
	plt.plot(x,x*slope + intercept)
	pred = []
	for i in Xf:
		pred.append(predictFromLine(slope, intercept,i))
	error = []
	for i in range(len(pred)):
		error.append((pred[i]-Xf[i])/Xf[i])
	plt.text(0.3,0.6, 'avg err = ' + str(round(np.abs(np.mean(error))*100))+ '%')
	plt.text(0.3,0.2, 'rsquared = ' + str('%.1g' % r_value**2))
	plt.text(0.4, 0.1	, 'p= ' + str(p_value))
	plt.xlabel('NCAA FT %')
	plt.ylabel('NBA eFG%')
	plt.show()

def fg():
	df = pd.read_csv('players.csv')
	df = df[['NBA_fg%','NCAA_ft']]
	X = df['NCAA_ft'].as_matrix()
	y = df['NBA_fg%'].as_matrix()
	Xf = []
	yf = []
	for i in range(0,len(X)): #strip invalid values
		if X[i] > 0 and X[i] < 1 and y[i] > 0 and y[i] < 1:
			Xf.append(X[i])
			yf.append(y[i])
	plt.figure(1)
	plt.scatter(Xf,yf, color = 'black', marker = '.')
	slope, intercept, r_value, p_value, std_err = stats.linregress(Xf,yf)
	x = np.linspace(0.3,0.9, 1000)
	plt.plot(x,x*slope + intercept)
	pred = []
	for i in Xf:
		pred.append(predictFromLine(slope, intercept,i))
	error = []
	for i in range(len(pred)):
		error.append((pred[i]-Xf[i])/Xf[i])
	plt.text(0.3,0.6, 'avg err = ' + str(round(np.abs(np.mean(error))*100))+ '%')
	plt.text(0.3,0.2, 'rsquared = ' + str('%.1g' % r_value**2))
	plt.text(0.4, 0.1	, 'p= ' + str('%.1g' % p_value))
	plt.xlabel('NCAA FT %')
	plt.ylabel('NBA FG%')
	plt.show()

def threept():
	df = pd.read_csv('players.csv')
	df = df[['NBA__3ptpct','NCAA_ft']]
	X = df['NCAA_ft'].as_matrix()
	y = df['NBA__3ptpct'].as_matrix()
	Xf = []
	yf = []
	for i in range(0,len(X)): #strip invalid values
		if X[i] > 0 and X[i] < 1 and y[i] > 0 and y[i] < 1:
			Xf.append(X[i])
			yf.append(y[i])
	plt.figure(1)
	plt.scatter(Xf,yf, color = 'black', marker = '.')
	slope, intercept, r_value, p_value, std_err = stats.linregress(Xf,yf)
	x = np.linspace(0.3,0.9, 1000)
	plt.plot(x,x*slope + intercept)

	pred = []
	for i in Xf:
		pred.append(predictFromLine(slope, intercept,i))
	error = []
	for i in range(len(pred)):
		error.append((pred[i]-Xf[i])/Xf[i])
	plt.text(0.3,0.6, 'avg err = ' + str(round(np.abs(np.mean(error))*100))+ '%')
	plt.text(0.3,0.4, 'rsquared = ' + str('%.1g' % r_value**2))
	plt.text(0.4, 0.1	, 'p= ' + str('%.1g' % p_value))
	plt.xlabel('NCAA FT %')
	plt.ylabel('NBA 3PT%')
	plt.show()


def multivar():
	df = pd.read_csv('players.csv')
	df = df[['NCAA_ft','NCAA_fgpct','NBA_efgpct']].dropna()
	lm = smf.ols(formula = 'NBA_efgpct ~ NCAA_ft + NCAA_fgpct ',data=df).fit()
	print(lm.summary(),lm.pvalues)
	

def multivar2():
	df = pd.read_csv('players.csv')
	df = df[['NCAA_ft','NCAA_fgpct','NBA_efgpct','position']].dropna()
	lm = smf.ols(formula = 'NBA_efgpct ~ NCAA_ft +position',data=df).fit()
	print(lm.summary(),lm.pvalues)

def straightUp():
	df = pd.read_csv('players.csv')
	df = df[['NCAA_ft','NCAA_fgpct','NBA_efgpct','NBA_ft%']].dropna()
	df = df.rename(columns = {'NBA_ft%': 'NBA_ft'})
	lm = smf.ols(formula = r'NBA_ft ~ NCAA_ft',data=df).fit()
	print(lm.summary(),lm.pvalues)
	intercept  = lm.params[0]
	slope= lm.params[1]
	#slope, intercept, r_value, p_value, std_err = stats.linregress(df['NCAA_ft'],df['NBA_ft'])
	#print(p_value,r_value**2)
	x = np.linspace(0,1, 1000)
	plt.plot(x,x*slope + intercept)
	plt.scatter(df['NBA_ft'],df['NCAA_ft'],color = 'black', marker = '.')
	plt.show()


def straightUpPos():
	df = pd.read_csv('players.csv')
	df = df[['NCAA_ft','NCAA_fgpct','NBA_efgpct','NBA_ft%', 'position']].dropna()
	df = df.rename(columns = {'NBA_ft%': 'NBA_ft'})
	lm = smf.ols(formula = r'NBA_ft ~ NCAA_ft + position',data=df).fit()
	print(lm.summary(),lm.pvalues)


	
#threept()
#fg()
#efgpct()
#multivar2()
straightUpPos()