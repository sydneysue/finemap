#!/usr/bin/env python
# coding: utf-8

#Sydney Sue - 260927733
#A2
#Should take around 30 mins to run in total

import pandas as pd
import numpy as np
import itertools
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

#Import data

zscore = pd.read_csv("/Users/sydneysue/Downloads/zscore.csv", index_col=0)
LDmatrix = pd.read_csv("/Users/sydneysue/Downloads/LD.csv", index_col=0)

#I made a list of only snps to trace back
snps = pd.read_csv("/Users/sydneysue/Downloads/snps.csv", header=None)

#Create an array of 100 numbers, representing the 100 SNPs
snp = np.arange(100)

#Get all the possible configurations with 1, 2 & 3 causal SNPs
one_config = list(itertools.combinations(snp,1))
two_config = list(itertools.combinations(snp,2))
three_config = list(itertools.combinations(snp,3))

#1 Implement the efficient Bayes factor for each causal configuration
#BF = N(zC|0,RCC+RCC*ΣCC*RCC)/N(zC|0,RCC)

#BF for one causal SNP
BF_one = []

for i in one_config:
    snp_i = snps.loc[i]
    x = zscore.loc[snp_i,'V1']
    R = LDmatrix.loc[snp_i,snp_i]
    sigma = 2.49 
    y = multivariate_normal.pdf(x, mean=0, cov=R+(R*sigma*R))
    z = multivariate_normal.pdf(x, mean=0, cov=R)
    BF = BF_one.append(y/z)

#BF for two causal SNPS - will take a long time to run 
BF_two = []

for i,j in two_config:
    snp_i = snps.loc[i]
    snp_j = snps.loc[j]
    
    #Create z-score array
    zscore_i = zscore.loc[snp_i,'V1'].values[0]
    zscore_j = zscore.loc[snp_j,'V1'].values[0]
    x = np.array([zscore_i,zscore_j])
    
    #Create the LD matrix
    LD_i = LDmatrix.loc[snp_i]
    LD_j = LDmatrix.loc[snp_j]
    LD_ij = pd.concat([LD_i,LD_j])
    LD_iij = LD_ij.loc[:,snp_i]
    LD_ijj = LD_ij.loc[:,snp_j]
    R = pd.concat([LD_iij, LD_ijj], axis=1, sort=False)
    R = R.to_numpy()
    
    sigma = np.array([[2.49,0], [0,2.49]])
    cov = (R+(np.dot(np.dot(R,sigma),R)))
    
    y = multivariate_normal.pdf(x, cov=cov, allow_singular=True)
    z = multivariate_normal.pdf(x, cov=R, allow_singular=True)
    BF = BF_two.append(y/z)    

#BF for three causal SNPS - will take a really long time to run 

BF_three = []

for i,j,k in three_config:
    snp_i = snps.loc[i]
    snp_j = snps.loc[j]
    snp_k = snps.loc[k]
    
    #Create z-score array
    zscore_i = zscore.loc[snp_i,'V1'].values[0]
    zscore_j = zscore.loc[snp_j,'V1'].values[0]
    zscore_k = zscore.loc[snp_k,'V1'].values[0]
    x = np.array([zscore_i,zscore_j,zscore_k])
    
    #Create LD matrix
    LD_i = LDmatrix.loc[snp_i]
    LD_j = LDmatrix.loc[snp_j]
    LD_k = LDmatrix.loc[snp_k]
    LD_ijk = pd.concat([LD_i,LD_j,LD_k])
    LD_iijk = LD_ijk.loc[:,snp_i]
    LD_ijjk = LD_ijk.loc[:,snp_j]
    LD_ijkk = LD_ijk.loc[:,snp_k]
    R = pd.concat([LD_iijk,LD_ijjk,LD_ijkk], axis=1, sort=False)
    R = R.to_numpy()
    
    sigma = np.array([[2.49,0,0], [0,2.49,0], [0,0,2.49]])
    cov = (R+(np.dot(np.dot(R,sigma),R)))
    
    y = multivariate_normal.pdf(x, cov=cov,allow_singular=True)
    z = multivariate_normal.pdf(x, cov=R,allow_singular=True)
    BF = BF_three.append(y/z)

#2 Implement the prior calculation for each configuration
#m is the number of SNPs in the locus
#k is the number of causal SNPs
m = 100
one_k = 1
two_k = 2
three_k = 3

one_prior = ((1/m)**one_k)*(((m-1)/m)**(m-one_k))
two_prior = ((1/m)**two_k)*(((m-1)/m)**(m-two_k))
three_prior = ((1/m)**three_k)*(((m-1)/m)**(m-three_k))

#3 Implement posterior inference on all possible causal configurations, assuming at max. 3 causal SNPs

#Calculate the sum of the posterior for all possible causal configuratiosn
one_sum = sum(BF_one)*one_prior
two_sum = sum(BF_two)*two_prior
three_sum = sum(BF_three)*three_prior
total_sum = one_sum + two_sum + three_sum

#Calculate normalized posterior for one causal SNP
one_post = []
for i in BF_one:
    one_post.append((i*one_prior)/total_sum)
    
#Calculate normalized posterior for two causal SNPs
two_post = []
for i in BF_two:
    two_post.append((i*two_prior)/total_sum)

#Calculate normalized posterior for three causal SNPs
three_post = []
for i in BF_three:
    three_post.append((i*three_prior)/total_sum)

#Join and sort lists to visualize
all_post = one_post + two_post + three_post
all_sorted = sorted(all_post)

x_axis = [x for x in range(len(all_sorted))]

plt.scatter(x_axis,all_sorted)
plt.title('Posterior of all configurations in increasing order')
plt.xlabel('Sorted configuration')
plt.ylabel('Configuration posterior')
plt.show()

#4 Implement PIP to calculate SNP-level posterior probabilities

total_snp_score = []
for i in range(0,100):
    snp_score = []
    #Get the index for each SNP in all possible configurations
    index_pos_one = [x for x, y in enumerate(one_config) if i in y]
    index_pos_two = [x for x, y in enumerate(two_config) if i in y]
    index_pos_three = [x for x, y in enumerate(three_config) if i in y]

    #Using the index, trace it to the normalized posteriors and sum to get SNP level posterior probabilities
    one_score = [one_post[i] for i in index_pos_one]
    snp_score.append(sum(one_score))
    
    two_score = [two_post[i] for i in index_pos_two]
    snp_score.append(sum(two_score))
    
    three_score = [three_post[i] for i in index_pos_three]
    snp_score.append(sum(three_score))
    
    total_snp_score.append(sum(snp_score))

#Plot PIP
x_axis = [x for x in range(len(total_snp_score))]

plt.scatter(x_axis,total_snp_score)
plt.title('SNP-level PIP results')
plt.xlabel('SNP')
plt.ylabel('PIP')
plt.show()

#Create snp_pip csv
pip = pd.DataFrame(total_snp_score, columns=["PIP"])

snp_pip = pd.concat([snps, pip], axis=1, sort=False)

# snp_pip.to_csv(r"/Users/sydneysue/Desktop/snp_pip.csv", index=False)

