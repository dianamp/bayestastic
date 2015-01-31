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
        self.N = 0

    def update(self, trial, success):
        self.trials[trial] = self.trials[trial] + 1
        self.N += 1
        if (success):
            self.successes[trial] = self.successes[trial] + 1

    def sample(self):
        raise NotImplementedError( "Need to implement a bandit algorithm" )
  
class BayesianBandit( BanditAlgorithm ): 
    def sample(self):
        sampled_theta = []
        for i in range(self.num_bandits):
            #Construct beta distribution for posterior
            dist = beta(self.prior[0]+self.successes[i],
                        self.prior[1]+self.trials[i]-self.successes[i])
            sampled_theta += [ dist.rvs() ]
        # Return the index of the sample with the largest value
        return sampled_theta.index( max(sampled_theta) ) 

class UCB1( BanditAlgorithm ):
    def sample(self):
        if (self.N < self.num_bandits):
            return self.N;
        mean_upper_bound = []
        for i in range(self.num_bandits):
            mean_upper_bound += [self.successes[i] / float(self.trials[i]) + sqrt( 2*log(self.N) / self.trials[i]) ]
        return mean_upper_bound.index( max(mean_upper_bound) ) 

def run_experiment(bandit, slots):
    n = 1000
    for i in range(n):
        next_bandit = bandit.sample()
        result = slots[next_bandit].pull()
        bandit.update(next_bandit, result)

def compare_algs(slots):
    print slots

    bb = BayesianBandit(num_bandits = len(slots))
    run_experiment(bb, slots)
    print "\tbayesian bandit: ", bb.trials

    ucb1 = UCB1(num_bandits = len(slots))
    run_experiment(ucb1, slots)
    print "\tUCB1: ", ucb1.trials
