from multi_armed_bandit import *

# Test out the Bayesian strategy
slots = [Bandit(0.3), Bandit(0.4)]
compare_algs(slots)

slots = [Bandit(0.3), Bandit(0.4), Bandit(0.2)]
compare_algs(slots)

slots = [Bandit(0.5), Bandit(0.4)]
compare_algs(slots)

slots = [Bandit(0.5), Bandit(0.4), Bandit(0.2), Bandit(0.2), Bandit(0.6)]
compare_algs(slots)

# TODO: plot success over time with python port of ggplot2
# TODO: Create a strategy for maximizing reward
