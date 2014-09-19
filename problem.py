#!/usr/bin/python

from __future__ import division
import numpy as np
import pandas as pd
from pandas import DataFrame

import re
import math


## This function is used to create permutations of size r from the set array 
def permutations(array, r=None):
    # permutations([1,2,3], 2) --> (1,2) (1,3) (2,1) (2,3) (3,1) (3,2)
    pool = tuple(array)
    n = len(pool)
    r = n if r is None else r
    if r > n:
        return
    indices = range(n)
    cycles = range(n, n-r, -1)
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return

def format_team(x):
	return "(a_" + str(x[0]) + ", b_" + str(x[1]) + ")"

## Create a dictionary that we will letter use to create our schedule
## n is the number of layers
def scheduledictionary(n):
	## generate n number from 1 to n
	m = n + 1 
	array = range(1, m)
	## create an empty dictionary and initialise it appropriately
	dictionary = {}
	for elt in array:
		dictionary[str(elt)] = []
	#
	y = permutations(array, 2)
	## Extract our permutations and put them in the dictionary
	for elt in y:
		dictionary[str(elt[0])].append(elt[1])
	df = pd.DataFrame(columns=["Team 1", "Team 2"])	
	#
	if(len(dictionary)):
		for i in range(len(dictionary['1'])):
			team = [1, dictionary['1'][i]]
			for k in range(2, m):
				x = dictionary[str(k)]
				for j in x:
					df.loc[df.shape[0]] = [format_team(team) , format_team([k, j])]
	#
	return df


def isswap(a, b):
	if ((a[0] == b[1]) and (a[1] == b[0])):
		return True
	else:
		return False




def schedule(n):
	m = n + 1
	array = range(1, m)
	dictionary = {}
	'''
	Get all permutations of size two!
	and put them in a list!
	We also, create a dictionary that we use a bit vector to check whether 
	two players have already played together. Keys of the dictionary are teams, and values associated 
	to all keys are initially set to zero. It changes to one once players of the corresponding keys involve in a match.
	'''
	y = permutations(array, 2)
	tab = []
	for elt in y:
		tab.append(elt)
		dictionary[str(elt)] = 0
	## Create a data frame that will contain our final schedule
	df = pd.DataFrame(columns=["team1", "team2"])
	for i in range(len(tab)):
		if(dictionary[str(tab[i])] == 0):
			dictionary[str(tab[i])] = 1 ## they have not played together yet, find their opponent 
			for j in range(len(tab)):
				if dictionary[str(tab[j])] == 0: ## they have not played together yet
					if(tab[i][0] != tab[j][0] and tab[i][1] != tab[j][1]):
						## check whether the players ids are swaped before proceeding forward
						## for example, (a_1, b_2) and (a_2, b_1) are swaped
						if(isswap(tab[i], tab[j])): 
							df.loc[df.shape[0]] = [format_team(tab[i]) , format_team(tab[j])]
							dictionary[str(tab[j])] = 1
							break;
	return df






			
		
