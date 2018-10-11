import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, SoupStrainer
import sys
import string
import requests
#use - takes input of player bball ref URL
def loadNBAPlayerPage(playerURL):
	page_request = requests.get(playerURL)
	soup = BeautifulSoup(page_request.text,"lxml")
	print(soup)

loadNBAPlayerPage('https://www.basketball-reference.com/players/b/bryanko01.html')