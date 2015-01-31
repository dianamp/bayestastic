from multi_armed_bandit import *

slots = [Bandit(0.3), Bandit(0.4)]
bandit = BayesianBandit()
run_experiment(bandit, slots)

slots = [Bandit(0.3), Bandit(0.4), Bandit(0.2)]
bandit = BayesianBandit(num_bandits=3)
run_experiment(bandit, slots)

slots = [Bandit(0.5), Bandit(0.4)]
bandit = BayesianBandit()
run_experiment(bandit, slots)
