import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels as sm
import matplotlib
import matplotlib.pyplot as plt
import markov_tests
import markov_generators as generators

data = None

def raw_input(s) :
    return input(s)

option = int(raw_input(" 1] Test a random numbers generator. \n 2] Find a data set's distribution. \n Choose an option : "))
if option == 1 :
    generator = int(raw_input(" 1] uniform.\n 2] .\n 3] .\n Choose a generator : "))
    u = st.uniform.rvs(size = 8000)
    if generator == 1 :
        data = generators.uniform_generator(u)
    elif generator == 2 :
        pass                        
    elif generator == 3 :
        pass
    else:
        print("invalid choice"); exit()
elif option == 2 :
    choice = int(raw_input(" 1] Take 'sea temp' example. \n 2] Take 'co2 ppvm' example. \n 3] Take 'sunspot activity' example \n 4] Load a file (format : observation1,observation2,...). \n Choose an option : "))
    if choice == 1 :
        # Load data from statsmodels datasets
        data = sm.datasets.elnino.load_pandas().data.set_index('YEAR').values.ravel()
    elif choice == 2 :
        data = sm.datasets.co2.load_pandas().data.values.ravel()
        data = data[data > 0]
    elif choice == 3 :
        data = sm.datasets.sunspots.load_pandas().data.values.ravel()
    elif choice == 4 :
        pass
    else :
        print("invalid choice"); exit()
else :
    print("invalid choice"); exit()

data = pd.Series(data)
# Plot for comparison
plt.figure(figsize=(12,8))
ax = data.plot(kind='hist', bins=50, normed=True, alpha=0.5)
plt.show()


# Find best fit distribution "SSE"
bestFitted = markov_tests.best_sse_fit_distributions(data, plt, 200)

# Find best fit distribution "q-qPlot" "chiSquare" "Kolmogorov-Smirnov"
markov_tests.qq_chiSquare_ks_Tests(data, bestFitted, plt)