import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt

print("================  TCL Test  ================")
instances = 15
instancesSize = 2000

distribution = st.uniform

randomNumbers = distribution.rvs(size=instances*instancesSize)

sse = []
def get_sse(size):
    normToTest = [0] * instancesSize
    for t in range(size) :
        tab = randomNumbers[(t) * instancesSize : (t + 1) * instancesSize]
        normToTest = (list(map(lambda x : x * t, normToTest)) + tab)/(t+1)
        y, x = np.histogram(normToTest, bins=50, density=True)
        data = pd.Series(normToTest)
        params = st.norm.fit(data)
        print(params)
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        norm = st.norm.pdf(x, loc=loc, scale=scale, *arg)
        sse.append(np.sum(np.power(norm[:-1] - y, 2.0)))
        
        plt.plot(x,norm)
        ax = data.plot(kind='hist', bins=50, normed=True, alpha=0.5)
            
        ax.set_title("GRAPH : sum of {} instance of {}".format(t+1,distribution.name))
        plt.show()
    return normToTest
get_sse(instances)

plt.figure(figsize=(12,8))
data = pd.Series(sse)
ax = data.plot()
ax.set_title("SSE GRAPH")
ax.set_xlabel("instances number")
ax.set_ylabel('sse')
plt.show()
