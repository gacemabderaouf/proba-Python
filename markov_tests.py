import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
from itertools import accumulate


# Compare "data" with a large set of known distributions to determine the maximum likelihood estimates 
def best_sse_fit_distributions(data, plt, bins=100):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data (used as parameter for "scipy.fit" function)
    y, x = np.histogram(data, bins=bins, density=True)
    
    #ajust 'x' axis (sincerely, I don't know why and/or how this shitty line does the job)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [        
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    ]
        # Best holders
    best_holders = [{"sse":np.inf}] * 5

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit (don't care about this)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data) #params are : [<shapes>, loc, scale]

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF (getting the theorical ProbabilityDistributionFunction) and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))


                # identify if this distribution is better
                if best_holders[4]["sse"] > sse :
                    i = 0
                    while 1 :
                        if best_holders[i]["sse"] < sse :
                            i += 1
                        else :
                            break
                    best_holders.insert(i,{"distribution":distribution,"params":params,"expectedPDF":pdf,"observedPDF":y,"sse":sse,"qq":0,"chi":0,"ks":0})
                    best_holders = best_holders[:-1]
        except Exception as e:
            print(e)
                
    plt.figure(figsize=(12,8))
    for dist in best_holders :
        param_names = (dist["distribution"].shapes + ', loc, scale').split(', ') if len(dist["params"]) > 2 else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, dist["params"])])
        dist_str = '{}({})'.format(dist["distribution"].name, param_str)
        pd.Series(dist["expectedPDF"],x).plot(label=dist_str, legend=True)
        print(dist_str)
    
    ax = data.plot(kind='hist', bins=50, normed=True, alpha=0.5)
            
    # Update plots
    param_names = (best_holders[0]["distribution"].shapes + ', loc, scale').split(', ') if len(best_holders[0]["params"]) > 2 else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_holders[0]["params"])])
    dist_str = '{}({})'.format(best_holders[0]["distribution"].name, param_str)
    
    
    ax.set_title("Best fit distribution 'SSE' : \n" + dist_str)
    ax.set_ylabel('Frequency')
    

    plt.show()
    return best_holders

def qq_chiSquare_ks_Tests(data, toTest, plt) :
    i = 0
    ax = [None] * len(toTest)
    summary =[]
    for dist in toTest :
        p = "23" + str(i)
        ax[i] = plt.subplot(p)
        ax[i].set_title("Q-Q plot : " + dist["distribution"].name)
        i += 1
        qq = st.probplot(data, sparams = dist["params"], dist=dist["distribution"], fit = True, plot=plt)
        print(qq[1])
        dist["qq"] = qq[1][-1] 
        chi = st.chisquare(dist["observedPDF"],dist["expectedPDF"])
        print(chi)
        dist["chi"] = chi[0]
        ks = st.kstest(data,dist["distribution"].name,args = dist["params"])
        print(ks)
        dist["ks"] = ks[0]
        
        summary.append([dist["sse"],dist["qq"],dist["chi"],dist["ks"]])

    summary = pd.DataFrame(summary, index = list(map(lambda x : x["distribution"].name,toTest)), columns = ["sse","qq","chi2","ks"])
    print(summary)
    plt.show()


