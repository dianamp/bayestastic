from numpy import *
from scipy.stats import beta,bernoulli

class Bandit(object):
    def __init__(self,p):
        self.p = p
    def pull(self):
        return bernoulli.rvs(self.p)

# This is adapted from https://gist.github.com/stucchio/5383149#file-beta_bandit-py
class BanditAlgorithm(object):
    def __init__(self, num_bandits=2):
        self.trials = zeros(shape=(num_bandits,), dtype=int)
        self.successes = zeros(shape=(num_bandits,), dtype=int)
        self.num_bandits = num_bandits
        self.prior = (1.0,1.0)

    def update(self, trial, success):
        self.trials[trial] = self.trials[trial] + 1
        if (success):
            self.successes[trial] = self.successes[trial] + 1

    def pick_bandit(self):
        raise NotImplementedError( "Need to implement a bandit algorithm" )
  
class BayesianBandit( BanditAlgorithm ): 
    def pick_bandit(self):
        sampled_theta = []
        for i in range(self.num_bandits):
            #Construct beta distribution for posterior
            dist = beta(self.prior[0]+self.successes[i],
                        self.prior[1]+self.trials[i]-self.successes[i])
            sampled_theta += [ dist.rvs() ]
        # Return the index of the sample with the largest value
        return sampled_theta.index( max(sampled_theta) ) 

# Experiment 1: Bayesian Bandit vs. UCB1 
# Chart # successes over time for iterations 1...1000
# Two bandits, 0.3, 0.4
def run_experiment(bandit, slots):
    n = 1000
    for i in range(n):
        next = bandit.pick_bandit()
        pull = slots[next].pull()
        bandit.update(next, pull)
    print bandit.trials
