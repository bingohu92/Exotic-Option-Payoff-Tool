import Tkinter
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import time 
import tkMessageBox

#Black and Scholes
def d1(S0, K, r, sigma, T):
    return (np.log(S0/K) + (r + sigma**2 / 2) * T)/(sigma * np.sqrt(T))
 
def d2(S0, K, r, sigma, T):
    return (np.log(S0 / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))
 
def BlackScholes(type,S0, K, r, sigma, T):
    if type=="C":
        return S0 * ss.norm.cdf(d1(S0, K, r, sigma, T)) - K * np.exp(-r * T) * ss.norm.cdf(d2(S0, K, r, sigma, T))
    else:
       return K * np.exp(-r * T) * ss.norm.cdf(-d2(S0, K, r, sigma, T)) - S0 * ss.norm.cdf(-d1(S0, K, r, sigma, T))


class Stock:
    def __init__(self,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
        self.S0 = S0
        self.K = K
        self.sigma = sigma
        self.r = r
        self.q = q
        self.T = T
        self.dt = dt
        self.max = S0
        self.min = S0
        self.average = S0
        self.mPrice = S0

    def simulation(self):
        N = round(self.T/self.dt)
        t = np.linspace(0, self.T, N)
        W = np.random.standard_normal(size = N) 
        W = np.cumsum(W)*np.sqrt(self.dt) ### standard brownian motion ###
        X = (self.r - self.q -0.5*self.sigma**2)*t + self.sigma*W 
        S = self.S0*np.exp(X) ### geometric brownian motion ###
        #plt.plot(t, S)
        #print(max(S))
        #plt.show()
        self.max = max(S)
        self.min = min(S)
        self.average = np.mean(S)
        self.mPrice = S[-1]
        self.Xmounth = 0

    def simulation_chooserOption(self,Xmounth):
        N = round(self.T/self.dt)
        t = np.linspace(0, self.T, N)
        W = np.random.standard_normal(size = N) 
        W = np.cumsum(W)*np.sqrt(self.dt) ### standard brownian motion ###
        X = (self.r - self.q -0.5*self.sigma**2)*t + self.sigma*W 
        S = self.S0*np.exp(X) ### geometric brownian motion ###
        self.mPrice = S[-1]
        return S[max(int(Xmounth/self.T/12 * len(S)) - 1, 0)]

    def get_maxPrice(self):
        return self.max

    def get_minPrice(self):
        return self.min

    def get_averagePrice(self):
        return self.average

    def get_maturityPrice(self):
        return self.mPrice


def binary_asset_or_nothing_call(STime=10000,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        MyPayoff = 0
        if (s.get_maturityPrice() > K):
            MyPayoff = s.get_maturityPrice()
        Payoffs.append(MyPayoff)
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Binary Asset-or-Nothing Call'+'\n'+'Strike='+str(K))
    plt.show()

def binary_asset_or_nothing_put(STime=10000,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    axes = plt.gca()
    axes.set_ylim([0,2 * K])
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        MyPayoff = 0
        if (s.get_maturityPrice() < K):
            MyPayoff = s.get_maturityPrice()
        Payoffs.append(MyPayoff)
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Binary Asset-or-Nothing Put'+'\n'+'Strike='+str(K))
    plt.show()

def binary_cash_or_nothing_call(STime=10000,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    axes = plt.gca()
    axes.set_ylim([0,2.0])
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        MyPayoff = 0
        if (s.get_maturityPrice() > K):
            MyPayoff = 1.0
        Payoffs.append(MyPayoff)
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Binary Cash-or-Nothing Call'+'\n'+'Strike='+str(K))
    plt.show()

def binary_cash_or_nothing_put(STime=10000,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    axes = plt.gca()
    axes.set_ylim([0,2.0])
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        MyPayoff = 0
        if (s.get_maturityPrice() < K):
            MyPayoff = 1.0
        Payoffs.append(MyPayoff)
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Binary Cash-or-Nothing Put'+'\n'+'Strike='+str(K))
    plt.show()

def lookback_fixedStrike_call(STime=10000,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        Payoffs.append(max(0,s.get_maxPrice()-K))
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Fixed-strike Lookback Call'+'\n'+'Strike='+str(K))
    plt.show()

def lookback_fixedStrike_put(STime=10000,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        Payoffs.append(max(0,K-s.get_minPrice()))
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Fixed-strike Lookback Put'+'\n'+'Strike='+str(K))
    plt.show()

def lookback_floatingStrike_call(STime=10000,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        Payoffs.append(max(0,s.get_maturityPrice()-s.get_minPrice()))
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Floating-strike Lookback Call'+'\n'+'Strike='+str(K))
    plt.show()

def lookback_floatingStrike_put(STime=10000,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        Payoffs.append(max(0,s.get_maxPrice()-s.get_maturityPrice()))
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Floating-strike Lookback Put'+'\n'+'Strike='+str(K))
    plt.show()

def asian_fixedStrike_call(STime=10000,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        Payoffs.append(max(0,s.get_averagePrice()-K))
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Asian Fixed-strike Lookback Call'+'\n'+'Strike='+str(K))
    plt.show()

def asian_fixedStrike_put(STime=10000,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        Payoffs.append(max(0,K-s.get_averagePrice()))
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Asian Fixed-strike Lookback Put'+'\n'+'Strike='+str(K))
    plt.show()

def asian_floatingStrike_call(STime=10000,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        Payoffs.append(max(0,s.get_maturityPrice()-s.get_averagePrice()))
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Asian Floating-strike Lookback Call'+'\n'+'Strike='+str(K))
    plt.show()

def asian_floatingStrike_put(STime=10000,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        Payoffs.append(max(0,s.get_averagePrice()-s.get_maturityPrice()))
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Asian Floating-strike Lookback Put'+'\n'+'Strike='+str(K))
    plt.show()

def barrier_up_and_in_call(STime=10000,barrier=150,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        myPayoff = 0
        if (s.get_maxPrice() > barrier):
            myPayoff = max(s.get_maturityPrice()-K,0)
        Payoffs.append(myPayoff)
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Barrier Up-and-In Call'+ '\n'+ 'Barrier=' + str(barrier) + '   Strike=' + str(K))
    plt.show()

def barrier_up_and_out_call(STime=10000,barrier=150,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        myPayoff = 0
        if (s.get_maxPrice() < barrier):
            myPayoff = max(s.get_maturityPrice()-K,0)
        Payoffs.append(myPayoff)
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Barrier Up-and-Out Call'+ '\n'+ 'Barrier=' + str(barrier) + '   Strike=' + str(K))
    plt.show()

def barrier_down_and_in_call(STime=10000,barrier=150,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        myPayoff = 0
        if (s.get_minPrice() < barrier):
            myPayoff = max(s.get_maturityPrice()-K,0)
        Payoffs.append(myPayoff)
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Barrier Down-and-In Call'+ '\n'+ 'Barrier=' + str(barrier) + '   Strike=' + str(K))
    plt.show()

def barrier_down_and_out_call(STime=10000,barrier=150,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        myPayoff = 0
        if (s.get_minPrice() > barrier):
            myPayoff = max(s.get_maturityPrice()-K,0)
        Payoffs.append(myPayoff)
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Barrier Down-and-Out Call'+ '\n'+ 'Barrier=' + str(barrier) + '   Strike=' + str(K))
    plt.show()

def barrier_up_and_in_put(STime=10000,barrier=100,S0=100,K=150,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        myPayoff = 0
        if (s.get_maxPrice() > barrier):
            myPayoff = max(K-s.get_maturityPrice(),0)
        Payoffs.append(myPayoff)
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Barrier Up-and-In Put'+ '\n'+ 'Barrier=' + str(barrier) + '   Strike=' + str(K))
    plt.show()

def barrier_up_and_out_put(STime=10000,barrier=100,S0=100,K=150,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        myPayoff = 0
        if (s.get_maxPrice() < barrier):
            myPayoff = max(K-s.get_maturityPrice(),0)
        Payoffs.append(myPayoff)
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Barrier Up-and-Out Put'+ '\n'+ 'Barrier=' + str(barrier) + '   Strike=' + str(K))
    plt.show()

def barrier_down_and_in_put(STime=10000,barrier=100,S0=100,K=150,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        myPayoff = 0
        if (s.get_minPrice() < barrier):
            myPayoff = max(K-s.get_maturityPrice(),0)
        Payoffs.append(myPayoff)
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Barrier Down-and-In Put'+ '\n'+ 'Barrier=' + str(barrier) + '   Strike=' + str(K))
    plt.show()

def barrier_down_and_out_put(STime=10000,barrier=100,S0=100,K=150,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        myPayoff = 0
        if (s.get_minPrice() > barrier):
            myPayoff = max(K-s.get_maturityPrice(),0)
        Payoffs.append(myPayoff)
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Barrier Down-and-Out Put'+ '\n'+ 'Barrier=' + str(barrier) + '   Strike=' + str(K))
    plt.show()

def chooser_option(time = 6,STime=10000,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        temp = s.simulation_chooserOption(time)
        Uprices.append(s.get_maturityPrice())
        if (temp > K):
            Payoffs.append(max(s.get_maturityPrice()-K,0))
        else:
            Payoffs.append(max(K-s.get_maturityPrice(),0))
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Chooser Option'+ '\n'+ str(time)+'-month into ' + str(T * 12 - time) + '-month chooser  Strike=' + str(K))
    plt.show()

def range_option(STime=10000,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        Payoffs.append(s.get_maxPrice() - s.get_minPrice())
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Range Option'+ '\n'+'  Strike=' + str(K))
    plt.show()

def power_call(power = 2,STime=10000,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        Payoffs.append(pow(max(s.get_maturityPrice()-K,0),power))
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Power Call'+ '\n'+ 'Power=' + str(power) + '  Strike=' + str(K))
    plt.show()

def power_put(power = 2,STime=10000,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        Payoffs.append(pow(max(K-s.get_maturityPrice(),0),power))
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Power Put'+ '\n'+ 'Power=' + str(power) + '  Strike=' + str(K))
    plt.show()

def restrike_call(restrike_boundary=85,restrike_strike = 75,STime=10000,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        temp = K
        if (s.get_minPrice() < restrike_boundary):
            temp = restrike_strike
        Payoffs.append(max(s.get_maturityPrice()-temp,0))
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Restrike Call'+ '\n'+ 'Restrike Boundary=' + str(restrike_boundary) + '  Restrike Strike=' + str(restrike_strike)  + '  Strike=' + str(K))
    plt.show()

def restrike_put(restrike_boundary=135,restrike_strike = 120,STime=10000,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        temp = K
        if (s.get_maxPrice() > restrike_boundary):
            temp = restrike_strike
        Payoffs.append(max(temp - s.get_maturityPrice(),0))
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Restrike Put'+ '\n'+ 'Restrike Boundary=' + str(restrike_boundary) + '  Restrike Strike=' + str(restrike_strike)  + '  Strike=' + str(K))
    plt.show()

def compound_COC(STime=10000,K1=7.5,T1=2,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T1-T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        Payoffs.append(max(BlackScholes("C",s.get_maturityPrice(),K,r,sigma,T) - K1, 0))
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Compound Option - Call on Call'+ '\n'+'Strike1=' + str(K)+'  Strike2=' + str(K1))
    plt.show()

def compound_POC(STime=10000,K1=7.5,T1=2,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T1-T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        Payoffs.append(max(BlackScholes("P",s.get_maturityPrice(),K,r,sigma,T) - K1, 0))
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Compound Option - Put on Call'+ '\n'+ 'Strike1=' + str(K)+'  Strike2=' + str(K1))
    plt.show()

def compound_COP(STime=10000,K1=7.5,T1=2,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T1-T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        Payoffs.append(max(K1-BlackScholes("C",s.get_maturityPrice(),K,r,sigma,T), 0))
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Compound Option - Call on Put'+ '\n'+'Strike1=' + str(K)+'  Strike2=' + str(K1))
    plt.show()

def compound_POP(STime=10000,K1=7.5,T1=2,S0=100,K=100,sigma=0.20,r=0,q=0,T=1,dt=0.001):
    Uprices = []
    Payoffs = []
    for i in range(0,STime):
        s = Stock(S0,K,sigma,r,q,T1-T,dt)
        s.simulation()
        Uprices.append(s.get_maturityPrice())
        Payoffs.append(max(K1-BlackScholes("P",s.get_maturityPrice(),K,r,sigma,T), 0))
    plt.plot(Uprices,Payoffs,marker='o',linestyle='',color='b')
    plt.xlabel('Underlying price')
    plt.ylabel('Payoff')
    plt.title('Payoff Diagram of Compound Option - Put on Put'+ '\n'+'Strike1=' + str(K)+'  Strike2=' + str(K1))
    plt.show()

form = Tkinter.Tk()
form.wm_title("Bingo's Exotic Option Payoff Tool")
var = Tkinter.IntVar()
Options_area = Tkinter.LabelFrame(form, text="Select exotic option here: ")
Options_area.grid(row=10, columnspan=3, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)

Binary_AON_Call = Tkinter.Radiobutton(Options_area, text="Binary Asset-or-Nothing Call", width=35,variable=var,value=1)
Binary_AON_Call.grid(row=0, column=0, sticky='W', padx=5, pady=2)

Binary_AON_Put = Tkinter.Radiobutton(Options_area, text="Binary Asset-or-Nothing Put", width=35,variable=var,value=2)
Binary_AON_Put.grid(row=1, column=0, sticky='W', padx=5, pady=2)

Binary_CON_Call = Tkinter.Radiobutton(Options_area, text="Binary Cash-or-Nothing Call",width=35,variable=var,value=3)
Binary_CON_Call.grid(row=2, column=0, sticky='W', padx=5, pady=2)

Binary_CON_Put = Tkinter.Radiobutton(Options_area, text="Binary Cash-or-Nothing Put", width=35,variable=var,value=4)
Binary_CON_Put.grid(row=3, column=0, sticky='W', padx=5, pady=2)

Compound_ConC = Tkinter.Radiobutton(Options_area, text="Compound Options: Call on a Call", width=35,variable=var,value=5)
Compound_ConC.grid(row=4, column=0, sticky='W', padx=5, pady=2)

Compound_ConP = Tkinter.Radiobutton(Options_area, text="Compound Options: Call on a Put", width=35,variable=var,value=6)
Compound_ConP.grid(row=5, column=0, sticky='W', padx=5, pady=2)

Compound_PonC = Tkinter.Radiobutton(Options_area, text="Compound Options: Put on a Call", width=35,variable=var,value=7)
Compound_PonC.grid(row=6, column=0, sticky='W', padx=5, pady=2)

Compound_PonP = Tkinter.Radiobutton(Options_area, text="Compound Options: Put on a Put", width=35,variable=var,value=8)
Compound_PonP.grid(row=7, column=0, sticky='W', padx=5, pady=2)

chooser_option = Tkinter.Radiobutton(Options_area, text="Chooser Option", width=35,variable=var,value=9)
chooser_option.grid(row=8, column=0, sticky='W', padx=5, pady=2)

range_option = Tkinter.Radiobutton(Options_area, text="Range Option", width=35,variable=var,value=10)
range_option.grid(row=9, column=0, sticky='W', padx=5, pady=2)

Lookback_FS_Call = Tkinter.Radiobutton(Options_area, text="Lookback Fixed-Strike Call", width=35,variable=var,value=11)
Lookback_FS_Call.grid(row=0, column=1, sticky='W', padx=5, pady=2)

Lookback_FS_Put = Tkinter.Radiobutton(Options_area, text="Lookback Fixed-Strike put", width=35,variable=var,value=12)
Lookback_FS_Put.grid(row=1, column=1, sticky='W', padx=5, pady=2)

Lookback_FloatS_Call = Tkinter.Radiobutton(Options_area, text="Lookback Floating-Strike Call", width=35,variable=var,value=13)
Lookback_FloatS_Call.grid(row=2, column=1, sticky='W', padx=5, pady=2)

Lookback_FloatS_put = Tkinter.Radiobutton(Options_area, text="Lookback Floating-Strike put", width=35,variable=var,value=14)
Lookback_FloatS_put.grid(row=3, column=1, sticky='W', padx=5, pady=2)

Asian_FS_Call = Tkinter.Radiobutton(Options_area, text="Asian Fixed-Strike Call", width=35,variable=var,value=15)
Asian_FS_Call.grid(row=4, column=1, sticky='W', padx=5, pady=2)

Asian_FS_Put = Tkinter.Radiobutton(Options_area, text="Asian Fixed-Strike Put", width=35,variable=var,value=16)
Asian_FS_Put.grid(row=5, column=1, sticky='W', padx=5, pady=2)

Asian_FloatS_Call = Tkinter.Radiobutton(Options_area, text="Asian Floating-Strike Call", width=35,variable=var,value=17)
Asian_FloatS_Call.grid(row=6, column=1, sticky='W', padx=5, pady=2)

Asian_FloatS_Put = Tkinter.Radiobutton(Options_area, text="Asian Floating-Strike Put", width=35,variable=var,value=18)
Asian_FloatS_Put.grid(row=7, column=1, sticky='W', padx=5, pady=2)

Power_Call = Tkinter.Radiobutton(Options_area, text="Power Call", width=35,variable=var,value=19)
Power_Call.grid(row=8, column=1, sticky='W', padx=5, pady=2)

Power_Put = Tkinter.Radiobutton(Options_area, text="Power Put", width=35,variable=var,value=20)
Power_Put.grid(row=9, column=1, sticky='W', padx=5, pady=2)

BO_UAI_Call = Tkinter.Radiobutton(Options_area, text="Barrier Options: Up and In Call", width=35,variable=var,value=21)
BO_UAI_Call.grid(row=0, column=2, sticky='W', padx=5, pady=2)

BO_UAO_Call = Tkinter.Radiobutton(Options_area, text="Barrier Options: Up and Out Call", width=35,variable=var,value=22)
BO_UAO_Call.grid(row=1, column=2, sticky='W', padx=5, pady=2)

BO_DAI_Call = Tkinter.Radiobutton(Options_area, text="Barrier Options: Down and In Call", width=35,variable=var,value=23)
BO_DAI_Call.grid(row=2, column=2, sticky='W', padx=5, pady=2)

BO_DAO_Call = Tkinter.Radiobutton(Options_area, text="Barrier Options: Down and Out Call", width=35,variable=var,value=24)
BO_DAO_Call.grid(row=3, column=2, sticky='W', padx=5, pady=2)

BO_UAI_Put = Tkinter.Radiobutton(Options_area, text="Barrier Options: Up and In Put", width=35,variable=var,value=25)
BO_UAI_Put.grid(row=4, column=2, sticky='W', padx=5, pady=2)

BO_UAO_Put = Tkinter.Radiobutton(Options_area, text="Barrier Options: Up and Out Put", width=35,variable=var,value=26)
BO_UAO_Put.grid(row=5, column=2, sticky='W', padx=5, pady=2)

BO_DAI_Put = Tkinter.Radiobutton(Options_area, text="Barrier Options: Down and In Put", width=35,variable=var,value=27)
BO_DAI_Put.grid(row=6, column=2, sticky='W', padx=5, pady=2)

BO_DAO_Put = Tkinter.Radiobutton(Options_area, text="Barrier Options: Down and Out Put", width=35,variable=var,value=28)
BO_DAO_Put.grid(row=7, column=2, sticky='W', padx=5, pady=2)

Restrike_Call = Tkinter.Radiobutton(Options_area, text="Restrike Call", width=35,variable=var,value=29)
Restrike_Call.grid(row=8, column=2, sticky='W', padx=5, pady=2)

Restrike_Put = Tkinter.Radiobutton(Options_area, text="Restrike Put", width=35,variable=var,value=30)
Restrike_Put.grid(row=9, column=2, sticky='W', padx=5, pady=2)

Relevant_info = Tkinter.LabelFrame(form, text="Enter relevant info here: ")
Relevant_info.grid(row=3, columnspan=8, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)

S0_label = Tkinter.Label(Relevant_info, text="S0:")
S0_label.grid(row=0, column=0, sticky='E')

S0_entry = Tkinter.Entry(Relevant_info) 
S0_entry.grid(row=0, column=1, sticky='E')

Strike_label = Tkinter.Label(Relevant_info, text="Strike (K):")
Strike_label.grid(row=0, column=2, sticky='E')

Strike_entry = Tkinter.Entry(Relevant_info)
Strike_entry.grid(row=0, column=3, sticky='E')

Volatility_label = Tkinter.Label(Relevant_info, text="Volatility (sigma):")
Volatility_label.grid(row=0, column=4, sticky='E')

Volatility_entry = Tkinter.Entry(Relevant_info)
Volatility_entry.grid(row=0, column=5, sticky='E')

InterestRate_label = Tkinter.Label(Relevant_info, text="Interest Rate (r):")
InterestRate_label.grid(row=0, column=6, sticky='E')

InterestRate_entry = Tkinter.Entry(Relevant_info)
InterestRate_entry.grid(row=0, column=7, sticky='E')

Dividend_label = Tkinter.Label(Relevant_info, text="Dividend Yield (q):")
Dividend_label.grid(row=1, column=0, sticky='E')

Dividend_entry = Tkinter.Entry(Relevant_info)
Dividend_entry.grid(row=1, column=1, sticky='E')

Maturity_label = Tkinter.Label(Relevant_info, text="Maturity in Years (T):")
Maturity_label.grid(row=1, column=2, sticky='E')

Maturity_entry = Tkinter.Entry(Relevant_info)
Maturity_entry.grid(row=1, column=3, sticky='E')

STime_label = Tkinter.Label(Relevant_info, text="Simulation times:")
STime_label.grid(row=1, column=4, sticky='E')

STime_entry = Tkinter.Entry(Relevant_info)
STime_entry.grid(row=1, column=5, sticky='E')

Barrier_label = Tkinter.Label(Relevant_info, text="Barrier:")
Barrier_label.grid(row=1, column=6, sticky='E')

Barrier_entry = Tkinter.Entry(Relevant_info)
Barrier_entry.grid(row=1, column=7, sticky='E')

RestrikePrice_label = Tkinter.Label(Relevant_info, text="Restrike Price:")
RestrikePrice_label.grid(row=2, column=0, sticky='E')

RestrikePrice_entry = Tkinter.Entry(Relevant_info)
RestrikePrice_entry.grid(row=2, column=1, sticky='E')

RestrikeStrike_label = Tkinter.Label(Relevant_info, text="Restrike Strike:")
RestrikeStrike_label.grid(row=2, column=2, sticky='E')

RestrikeStrike_entry = Tkinter.Entry(Relevant_info)
RestrikeStrike_entry.grid(row=2, column=3, sticky='E')

Power_label = Tkinter.Label(Relevant_info, text="Power:")
Power_label.grid(row=2, column=4, sticky='E')

Power_entry = Tkinter.Entry(Relevant_info)
Power_entry.grid(row=2, column=5, sticky='E')

CMonth_label = Tkinter.Label(Relevant_info, text="Chooser Month:")
CMonth_label.grid(row=2, column=6, sticky='E')

CMonth_entry = Tkinter.Entry(Relevant_info)
CMonth_entry.grid(row=2, column=7, sticky='E')

CStrike2_label = Tkinter.Label(Relevant_info, text="Compound Strike2:")
CStrike2_label.grid(row=3, column=0, sticky='E')

CStrike2_entry = Tkinter.Entry(Relevant_info)
CStrike2_entry.grid(row=3, column=1, sticky='E')

CMaturity2_label = Tkinter.Label(Relevant_info, text="Compound Maturity2:")
CMaturity2_label.grid(row=3, column=2, sticky='E')

CMaturity2_entry = Tkinter.Entry(Relevant_info)
CMaturity2_entry.grid(row=3, column=3, sticky='E')

def setDault():
    clearInfo()
    S0_entry.insert(0,"100")
    Strike_entry.insert(0,"100")
    Volatility_entry.insert(0,"0.20")
    InterestRate_entry.insert(0,"0")
    Dividend_entry.insert(0,"0")
    Maturity_entry.insert(0,"1")
    STime_entry.insert(0,"10000")
    Barrier_entry.insert(0,"150")
    RestrikePrice_entry.insert(0,"85")
    RestrikeStrike_entry.insert(0,"75")
    Power_entry.insert(0,"2")
    CMonth_entry.insert(0,"6")
    CStrike2_entry.insert(0,"7.5")
    CMaturity2_entry.insert(0,"2")
    

def clearInfo():
    S0_entry.delete(0,Tkinter.END)
    Strike_entry.delete(0,Tkinter.END)
    Volatility_entry.delete(0,Tkinter.END)
    InterestRate_entry.delete(0,Tkinter.END)
    Dividend_entry.delete(0,Tkinter.END)
    Maturity_entry.delete(0,Tkinter.END)
    STime_entry.delete(0,Tkinter.END)
    Barrier_entry.delete(0,Tkinter.END)
    RestrikePrice_entry.delete(0,Tkinter.END)
    RestrikeStrike_entry.delete(0,Tkinter.END)
    Power_entry.delete(0,Tkinter.END)
    CMonth_entry.delete(0,Tkinter.END)
    CStrike2_entry.delete(0,Tkinter.END)
    CMaturity2_entry.delete(0,Tkinter.END)

def run():
    choice = var.get()
    if (choice == 1):
        binary_asset_or_nothing_call(int(STime_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 2):
        binary_asset_or_nothing_put(int(STime_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 3):
        binary_cash_or_nothing_call(int(STime_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 4):
        binary_cash_or_nothing_put(int(STime_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 5):
        compound_COC(int(STime_entry.get()),float(CStrike2_entry.get()),float(CMaturity2_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 6):
        compound_POC(int(STime_entry.get()),float(CStrike2_entry.get()),float(CMaturity2_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 7):
        compound_COP(int(STime_entry.get()),float(CStrike2_entry.get()),float(CMaturity2_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 8):
        compound_POP(int(STime_entry.get()),float(CStrike2_entry.get()),float(CMaturity2_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 9):
        chooser_option(int(CMonth_entry.get()),int(STime_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 10):
        range_option(int(STime_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 11):
        lookback_fixedStrike_call(int(STime_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 12):
        lookback_fixedStrike_put(int(STime_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 13):
        lookback_floatingStrike_call(int(STime_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 14):
        lookback_floatingStrike_put(int(STime_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 15):
        asian_fixedStrike_call(int(STime_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 16):
        asian_fixedStrike_put(int(STime_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 17):
        asian_floatingStrike_call(int(STime_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 18):
        asian_floatingStrike_put(int(STime_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 19):
        power_call(float(Power_entry.get()),int(STime_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 20):
        power_put(float(Power_entry.get()),int(STime_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 21):
        barrier_up_and_in_call(int(STime_entry.get()),float(Barrier_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 22):
        barrier_up_and_out_call(int(STime_entry.get()),float(Barrier_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 23):
        barrier_down_and_in_call(int(STime_entry.get()),float(Barrier_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 24):
        barrier_down_and_out_call(int(STime_entry.get()),float(Barrier_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 25):
        barrier_up_and_in_put(int(STime_entry.get()),float(Barrier_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 26):
        barrier_up_and_out_put(int(STime_entry.get()),float(Barrier_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 27):
        barrier_down_and_in_put(int(STime_entry.get()),float(Barrier_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 28):
        barrier_down_and_out_put(int(STime_entry.get()),float(Barrier_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 29):
        restrike_call(float(RestrikePrice_entry.get()),float(RestrikeStrike_entry.get()),int(STime_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    elif (choice == 30):
        restrike_put(float(RestrikePrice_entry.get()),float(RestrikeStrike_entry.get()),int(STime_entry.get()),float(S0_entry.get()),float(Strike_entry.get()),float(Volatility_entry.get()),float(InterestRate_entry.get()),float(Dividend_entry.get()),float(Maturity_entry.get()),dt=0.001)
    else:
        tkMessageBox.showerror("Warning", "Please select an option!")

buttion_Default = Tkinter.Button(Relevant_info, text="Default",command=setDault)
buttion_Default.grid(row=3, column=5)

buttion_Clear = Tkinter.Button(Relevant_info, text="Clear",command=clearInfo)
buttion_Clear.grid(row=3, column=6)

buttion_Run = Tkinter.Button(Relevant_info, text="Run",command=run)
buttion_Run.grid(row=3, column=7)
    

form.mainloop()
