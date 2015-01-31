from bayes_bandit import *



slots = [Bandit(0.3), Bandit(0.4)]
bandit = BetaBandit()
run_experiment(bandit, slots)
