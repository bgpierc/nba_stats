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
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.utils import shuffle
from scipy import stats
import seaborn as sns
import scipy.stats as st
import sklearn.linear_model as lm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def predictFromLine(slope, intercept,x):
	return slope*x + intercept
def NBAfgpct():
	df = pd.read_csv('players.csv')
	Xtrain = []
	ytrain = []
	Xtest = []
	ytest = []
	for i in range(0,int(len(df))):
		if df['NBA_fga_per_game'][i] > 0 and df['NCAA_ftapg'][i] > 0: #only include players that average >0 fg in NBA and >0 ft in NCAA
			if df['NCAA_ft'][i] >0 and df['NBA_fg%'][i] > 0: #eliminate invalid/corrupted values
				if df['NCAA_ft'][i] <1 and df['NBA_fg%'][i] < 1:
					Xtrain.append(df['NCAA_ft'][i])
					ytrain.append(df['NBA_fg%'][i])
	for i in range(int(len(df)/2),len(df)):
		if df['NBA_fga_per_game'][i] > 0 and df['NCAA_ft'][i] > 0:
			if df['NCAA_ft'][i] >0 and df['NBA_fg%'][i] > 0:
				if df['NCAA_ft'][i] <1 and df['NBA_fg%'][i] < 1:
					Xtest.append(df['NCAA_ft'][i])
					ytest.append(df['NBA_fg%'][i])

	x = np.linspace(0.2,1,100) #interval over which we evaluate regression line

	slope, intercept, r_value, p_value, std_err = stats.linregress(Xtrain,ytrain) #regression
	plt.figure(1)
	plt.scatter(Xtrain,ytrain,c = 'black', marker = '.')
	plt.plot(x,x*slope + intercept)
	plt.xlabel('NCAA ft %')
	plt.ylabel('NBA fg%')
	plt.text(0.3,0.2, 'rsquared = ' + str('%.1g' % r_value**2))

	plt.figure(2)

	pred = []
	for i in Xtest:
		pred.append(predictFromLine(slope,intercept,i)) #generate predicted values
	residuals = []
	for i in range(len(pred)):
		residuals.append(pred[i]-ytrain[i]) #find residuals
	plt.scatter(pred,residuals, c = 'black', marker = '.') #scatter plot
	slope, intercept, r_value, p_value, std_err = stats.linregress(pred,residuals)
	print(r_value**2)
	x2 = np.linspace(0.36,0.5,100)
	plt.plot(x2, x2*slope + intercept)
	plt.xlabel('Predicted NBA fg% (regression line)')
	plt.ylabel('Actual NBA fg%')
	plt.title('Residuals')
	plt.text(0.37,-0.2, 'rsquared = ' + str('%.1g' % r_value**2))
	plt.show()


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

def NBA3ptpct():
	df = pd.read_csv('players.csv')
	Xtrain = []
	ytrain = []
	Xtest = []
	ytest = []
	for i in range(0,int(len(df))):
		if df['NBA__3ptapg'][i] > 1 and df['NCAA_ftapg'][i] > 1:
			if df['NCAA_ft'][i] >0 and df['NBA__3ptpct'][i] > 0:
				if df['NCAA_ft'][i] <1 and df['NBA__3ptpct'][i] < 1:
					Xtrain.append(df['NCAA_ft'][i])
					ytrain.append(df['NBA__3ptpct'][i])
	for i in range(int(len(df)/2),len(df)):
		if df['NBA_fga_per_game'][i] > 0 and df['NCAA_ft'][i] > 0:
			if df['NBA__3ptapg'][i] > 0 and df['NBA__3ptapg'][i] > 0:
				if df['NCAA_ft'][i]<1 and df['NBA__3ptpct'][i] < 0.5 and df['NBA__3ptpct'][i] > 0:
					Xtest.append(df['NCAA_ft'][i])
					ytest.append(df['NBA__3ptpct'][i])
	Xtrain = np.asarray(Xtrain)
	ytrain = np.asarray(ytrain)
	Xtest = np.asarray(Xtest)
	ytest = np.asarray(ytest)


	plt.figure(1)
	x = np.linspace(0.2,1,100) #interval over which we evaluate regression line
	slope, intercept, r_value, p_value, std_err = stats.linregress(Xtrain,ytrain) #regression
	plt.scatter(Xtrain,ytrain,c = 'black', marker = '.')
	plt.plot(x,x*slope + intercept)
	plt.xlabel('NCAA ft %')
	plt.ylabel('NBA fg%')
	plt.text(0.3,0.2, 'rsquared = ' + str('%.1g' % r_value**2))

	plt.figure(2)

	pred = []
	for i in Xtrain:
		pred.append(predictFromLine(slope,intercept,i)) #generate predicted values
	residuals = []
	for i in range(len(pred)):
		residuals.append(pred[i]-ytrain[i]) #find residuals
	plt.scatter(pred,residuals, c = 'black', marker = '.') #scatter plot
	slope, intercept, r_value, p_value, std_err = stats.linregress(pred,residuals)
	print(r_value**2)
	x2 = np.linspace(0.29,0.40,1000)
	plt.plot(x2, x2*slope + intercept)
	plt.xlabel('Predicted NBA fg% (regression line)')
	plt.ylabel('Actual NBA fg%')
	plt.title('Residuals')
	plt.text(0.37,0.2, 'rsquared = ' + str('%.1g' % r_value**2))
	plt.show()
def calcEfgpct(NCAA_FGM,NCAA_3PTM,FGA):
	if(FGA != 0):
		return (NCAA_FGM +(0.5*NCAA_3PTM))/FGA
	else:
		return None
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
	plt.text(0.3,0.2, 'rsquared = ' + str('%.1g' % r_value**2))
	plt.xlabel('NCAA FT %')
	plt.ylabel('NBA eFG%')
	plt.show()




efgpct()