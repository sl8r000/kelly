import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy import stats, special
from scipy.stats import mstats

st.title("Information Theory for Investing")

st.markdown("""
In 1956, the mathematician John Kelly didn't just discover how to get rich quick — he also proved there was no way to get rich quicker. Given a favorable bet that you can make repeatedly, Kelly determined how much to wager each time so that you maximize the rate at which your wealth grows in the long run. Below, we'll see how Kelly's method works by starting with a simple coin flip example and gradually modifying it to get more interesting.
""")

st.header("Classical Coin Flip Example")

st.markdown("""
Let's start building intiution with a classical coin flip example. You have a biased coin that lands heads up with probability $p > 0.5$. You can bet as much as you want on a flip of that coin, with the following "double or nothing" payoffs:

* If it lands heads, you double whatever you bet;
* If it lands tails, you lose whatever you bet.

So if you bet \$5 and win, you'll get \$10 (your original \$5 plus \$5 more). If you bet \$5 and lose, your \$5 is gone. You can play repeatedly — as many times as you want, as long as you still have money to bet. 

So, how much should you bet? This is the question that Kelly answered in '56.

Here's how he reasoned about it. Let's constrain the problem for a moment, and say that you're going to bet the same, fixed fraction $f$ of your total wealth on each flip. So, for example, if $f=20\%$ and you start with \$100, then your first bet is \$20. If you win that first flip, your next bet will be \$24 (20% of \$120). If you lose, it'll be \$16. And so on. Kelly then asked: What is the optimal choice of $f$ for maximizing the growth rate of your wealth?

Before getting to the answer, let's play around with a simulation to get a feel for what happens as $p$ and $f$ vary. Below, you can choose the value of $p$ and see how different choices of $f$ fare over 10,000 coin flips. These are plotted on a log y scale with base 10, so a "50" on that axis means that you multiplied your wealth by $10^{50}$.
""")

p = st.slider(
    "Choose the value of p, the probability that the coin lands heads up.",
    min_value=0.0,
    max_value=1.0, 
    value=0.75
)
n_flips = 10000

def calc_log_return(f, flips):
    terms = np.clip(1 - f + 2*f*flips,1e-100,None)
    return np.log10(terms).cumsum()

def plot_flip_lines(p, n_flips):
    flips = np.random.binomial(1, p, size=n_flips)

    f_values = np.arange(1,10)*0.1 # Show some choices of f from 0 to 1
    df = pd.DataFrame({
        f'f={100*x:0.0f}%': calc_log_return(x,flips)
        for x in f_values
    })
    df.plot(
        title=f"Log Returns from 10,000 Coin Flips\nof a coin that lands heads up with probability {p:0.2f}",
        colormap="plasma")
    plt.xlabel("Coin Flips")
    plt.ylabel("Log Return")
    st.pyplot()

plot_flip_lines(p, n_flips)

st.markdown("""
You may have noticed that it's possible to bet too much, even though this is a favorable game. At the extreme end, if you bet $f=100\%$ of your wealth on each flip, then it only takes a single tails to wipe you out completely. So your probability of going bankrupt is $1-(1-p)^{n}$ which goes to $1$ as $n \\to \infty$. More generally, you can see that high values of $f$ tend to worse outcomes.

Let's work now to find the optimal value of $f$, which we'll denote as $f^*$. Once we write down the relevant equation, this will become a straightforward maximization problem. I'll write $X_k$ for the result of the $k^{\\text{th}}$ coin flip, where $X_k = 1$ if heads and $X_k = -1$ if tails. If you bet $f$ of your bankroll $n$ times, then your wealth at the end of those $n$ flips will be multiplied by
""")

st.latex("""
\prod_{k=1}^n (1 + fX_k) = (1 + f)^{H_n}(1 - f)^{T_n}.
""")

st.markdown("""
where $H_n$ is the number of heads from those $n$ flips and $T_n$ is the number of tails. Looking at this equation, Kelly chose to maximize the rate of return as a function of $f$, as $n \\to \infty$. In other words, we're choosing $f$ to maximize
""")

st.latex("""
\lim_{n\\to \infty} \sqrt[n]{(1 + f)^{H_n}(1 - f)^{T_n}}
""")


st.markdown("""
Notationally, it's a little easier to instead write
""")

st.latex("""

e^{rn} = (1 + f)^{H_n}(1 - f)^{T_n}

""")


st.markdown("""
and say that we're choosing $f$ to maximize $r$ as $n\\to \infty$. This amounts to the exact same thing, of course, but it's going to make the notation a little easier. (It also makes some connections to information theory clearer; we'll briefly note this later.) We'll call $r$ the "exponential rate of return".

You might ask why we'd want to maximize the rate of return. Intuitively, this rate is what will matter most to your wealth in the long run — starting with \$1,000 and compounding at 20% per annum for 100 years will leave you wealthier than starting with \$1,000,000 and compounding at 10% per annum.

In any case, let's solve this maximization problem. We take logs of each side to get that
""")

st.latex("""

r = \\frac{H_n}{n}\log(1+f) + \\frac{T_n}{n}\log(1-f).

""")

st.markdown("""
$H_n$ trends towards $np$ (probability of heads times the number of flips), so $H_n/n$ trends towards $p$. Or, formally, because $H_n \sim \\text{Binom}(n,p)$, so that $\mathbb{E}(H_n) = np$ we have that $H_n/n \\to p$ almost surely as $n \\to \infty$ by the strong law of large numbers. And similarly $T_n/n \\to 1-p$. So, in the limit as $n \\to \infty$, we're trying to maximize
""")

st.latex("""
p\log(1+f) + (1-p)\log(1-f).
""")

st.markdown("""
So set the deriviative with respect to $f$ to zero, and now we're just solving
""")

st.latex("""
\\frac{p}{1+f} = \\frac{1-p}{1-f}.
""")

st.markdown("""
And this equation is solved by $f^*=2p-1$. So this is the answer to our original question: When the probability of heads is $p$, you should bet $2p-1$ of your current wealth on each flip. This is more conservative than you might imagine: When $p=75\%$, for example, you only want to bet 50% of your wealth on each flip.

It's worth playing around with this for a little, to get a feel for what other values of $f$ look like. Below, you can set $f$ (with $p$ set by the slider above) to see what happens to your returns over time, and what happens to the equation $p\log(1+f) + (1-p)\log(1-f)$, i.e. the log EV of our return multiple. (If you want to change $p$ too, you can do it with the slider above, from the last plot.)
""")

fv = st.slider("Choose the value of f, the fraction of your bankroll to bet on each flip", min_value=0.0, max_value=1.0, value=0.75)

def plot_first_returns(fv, n_flips):
    flips = np.random.binomial(1, p, size=n_flips)
    returns = {}
    returns[f'Your choice of f={100*fv:0.0f}%'] = calc_log_return(fv,flips)
    returns[f'Optimal choice of f={100*(2*p-1):0.0f}%'] = calc_log_return(2*p-1,flips)
    df = pd.DataFrame(returns)

    fig, (ax1, ax2) = plt.subplots(1,2)

    df.plot( ax=ax1)
    ax1.set_title(f"Log Returns from 10,000\n Coin Flips of a coin that lands\nheads up with probability {p:0.2f}", fontsize=10)
    ax1.set_xlabel("Coin Flips")
    ax1.set_ylabel("Log Return")
    ax1.legend(fontsize=8)

    f = np.linspace(0, 1, 101)
    log_ev = p*np.log(1+f) + (1-p)*np.log(1-f)
    ax2.plot(f, log_ev, color='black')
    ax2.plot(2*p-1, p*np.log(1+2*p-1) + (1-p)*np.log(1 - 2*p+1), 'ro', color='tab:orange', label=f'Optimal choice of f={100*(2*p-1):0.0f}%')
    ax2.plot(fv, p*np.log(1+fv) + (1-p)*np.log(1 - fv), 'ro', color='tab:blue', label=f'Your choice of f={100*fv:0.0f}%')
    ax2.legend(fontsize=8)
    ax2.set_title("Value of\n$p\cdot\log(1+f) + (1-p)\cdot\log(1-f)$\nfor your choice vs the optimal one", fontsize=10)
    ax2.set_xlabel("Value of $f$")
    ax2.set_ylabel("Expected Log Return")
    plt.tight_layout()
    st.pyplot()

plot_first_returns(fv, n_flips) 

st.markdown("""
There's nothing special about the coin flip example, the same reasoning applies for any favorable game that you can play repeatedly. In general, suppose you have a bet $X$ with payoffs such that your wealth gets multiplied by $1+fX$ when you bet $f$ of your bankroll. You can make this bet repeatedly, using IID $X_k \sim X$. So by the same reasoning as above, your wealth multiple after playing $n$ times is
""")

st.latex("""
\\prod_{k=1}^n (1 + fX_k)
""")

st.markdown("""
and you want to maximize the rate of return $r$ in
""")

st.latex("""
e^{rn} = \\prod_{k=1}^n (1 + fX_k)
""")

st.markdown("""
as $n \\to \infty$, which leads you to
""")

st.latex("""
r = \\frac{1}{n}\sum_{k=1}^n \log(1+fX_k).
""")

st.markdown("""
As $n \\to \infty$, this goes to
""")

st.latex("""
\\mathbb{E}_X[\log(1+fX)]
""")

st.markdown("""
almost surely. So, just as with the coin flip example, we want to find $f^*$ to maximize the expected value of the log of our payout multiple. And this this is why you'll often hear Kelly betting described as "log optimal" betting.

A little bookkeeping: At the beginning of this section, we assumed that we were going to bet a fixed fraction $f$ of our wealth. But in general, might we not have to consider betting some sequence $f_k$ on each $X_k$? No, not if the $X_k$ are IID. The sketch of the argument is just that we'd get the same value of $f_k = f$ for each $k$, conditioning on the earlier outcomes. (Even when the $X_k$ are dependent on each other, it turns out that the strategy that maximizes your wealth growth rate is still to make the conditionally log-optimal bet.)

I also said it was obvious that the wealth growth multiple, $1+fX$, is the right thing to maximize, since this is what dominates returns in the long run. Actually, this is a bit "controversial", and the subject of a long-running feud in the literature, generally waged between a group of mathematicians who are pro-Kelly and a group of classical economists who are anti-Kelly. Econ Nobel Laureate Paul Samuelson even wrote a super [snarky
paper](http://www-stat.wharton.upenn.edu/~steele/Courses/434/434Context/Kelly%20Resources/Samuelson1979.pdf) arguing against the Kelly criterion, using mostly one-syllable words so that his oponents could understand him.

One can argue about utility functions, but I basically think the log-optimal crowd is right — especially if you're managing your own money. For a fun history of the Kelly criterion, check out William Poundstone's [Fortune's Formula](https://www.amazon.com/Fortunes-Formula-Scientific-Betting-Casinos/dp/0809045990). Ed Thorp, one of the mathematicians in the debate (and the inventor of card counting for blackjack) has a [great autobiography](https://www.amazon.com/Man-All-Markets-Street-Dealer/dp/0812979907), and compiled a [set of papers](https://www.amazon.com/KELLY-CAPITAL-GROWTH-INVESTMENT-CRITERION/dp/9814383139) with MacLean and Ziemba.. My favorite take on the economists vs mathematicians debate comes from Ole Peters in a [2012 talk](https://www.youtube.com/watch?v=f1vXAHGIpfc) at Gresham College, where he points out that the economists' perspective optimizes for an "ensemble average" over a population, while the mathematicians' perspective optimizes for an individual outcome over time. 
""")

st.header("Betting with an Edge")

st.markdown("""
As promised, let's now make our coin flip example a little more interesting. In many gambling settings, the odds on a bet are set by other participants — like [parimutuel](https://en.wikipedia.org/wiki/Parimutuel_betting) betting for horse races, pot odds in poker, or even the market pricing of securities in a financial market. To add this idea to our coin flip example, let's put in a toy version of the efficient market hypothesis. Let's say that, instead of the coin flip paying out at even odds, the odds are set by "the market": the market will "bid up" the price of the game until the expected gain is zero. Specifically, if "the market" "thinks" that the probability of heads is $p$, then the toy EMH says that market participants will quickly bid up the price of the game until the odds become $1/p$ times your money if heads (and you lose your money if tails, as before). This makes the EV of \$1 wagered equal to $p\cdot (1/p) + (1-p) \cdot 0 = 1$ (i.e. no expected gain).

If you think the market is right, i.e. you agree that the probability of heads is $p$, then you shouldn't bet at all. Kelly's criterion will tell you that $f^* = 0$ in this case. But what if you have an edge? What if you know that *real* probability of heads is $q$, even though the market thinks that it's $p$? Let's say that $q > p$, so that our edge makes us want to "go long" and pay the market price for the bet. How much should we bet? The reasoning is the same as before. The log EV of our wealth multiple is
""")

st.latex("""

\mathbb{E}_X[\log(1+fX)] = q\log(1+f(1/p-1)) + (1-q)\log(1-f)

""")

st.markdown("""

so we take derivatives as before, and get the equation

""")

st.latex("""

\\frac{q\\left(\\frac{1}{p}-1\\right)}{1 + f\\left(\\frac{1}{p}-1\\right)} = \\frac{1-q}{1-f}

""")

st.markdown("""

which is solved by $f^*=(q/p-1)/(1/p-1)$. And the exponential rate at which our wealth grows is

""")

st.latex("""

\mathbb{E}_X[\log(1+f^*X)] = q\log(q/p) + (1-q)\log\\left(\\frac{1-q}{1-p}\\right)

""")

st.markdown("""
So Kelly tells us how much we should bet on our edge, and how valuable that edge is to us.

The expression above might look familiar — it's the [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) from $p$ to $q$, where $q$ is the "real" distribution and $p$ is the "inefficient" one. This is a little surprising, because the [usual interpretation](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Motivation) of KL divergence is the number of extra bits it takes to encode a message drawn from some probability distribution $Q$ when you optimize your encoding for some other distribution $P$. So inefficient compression is related to "inefficiency" in the market, in the sense that there's an exploitable edge we can use to consistently make excess return. [Cover and Thomas](https://www.amazon.com/Thomas-M-Cover-Elements-Information/dp/B00HTK9U28), pp 171–175, have a great explanation of why this is true, showing how you can get optimal encoding from log optimal betting and vice versa.
""")

st.header("Bayesian Inference")

st.markdown("""
Now let's make another modification to our coin flip example. In a more realistic setting, we would't have access to the true probability of heads. We'd only be able to estimate it. Let's leave efficient markets aside for a moment, and return to even odds (double our money if heads, lose our money if tails). We watched somebody play before us, and saw that the coin landed heads 17 out of 20 times. Based on that, we think that the true probability of heads $p$ follows a [beta distribution](https://en.wikipedia.org/wiki/Beta_distribution), namely $p \sim
\\text{Beta}(18,4)$. (We have no prior information on $p$, so we think it's uniform before we observe any data.)
""")

def plot_beta_dist():

    x = np.linspace(0,1,101)
    s = pd.Series(
        stats.beta(18,4).pdf(x),
        index=x
    )
    s=s/s.sum()
    plt.title("Probability Density Function\nfor a Beta$(18,4)$ distribution")
    plt.ylabel("$\mathbb{P}(x)$")
    plt.xlabel("x")
    s.plot()
    st.pyplot()

plot_beta_dist()

st.markdown("""

Before, we assumed that we knew $p=85\%$, but now we only know that $p\sim\\text{Beta}(18,4)$. How will this affect the optimal choice of $f$? Let's get a little more general and write $a$ for the number of heads we observe, and $b$ for the number of tails. So when we have an uninformative prior for $p$, we think that $p \sim \\text{Beta}(a+1,b+1)$. As before, we want to maximize

""")

st.latex("""

\mathbb{E}[\\log(1 + fX)].

""")

st.markdown("""

As before, $X$ is a random variable that depends on $p$. But now that $p$ is a random variable too, we need to take the expected value over the joint distribution. We can do this by simply integrating the EV of $X$ conditional on $p$. In general, we might have to do this numerically. But in this case, we luck out and there is a simple closed formula. Let's see it. We calculate

""")

st.latex("""

\mathbb{E}_X[\\log(1 + fX)] = \int_{0}^{1} \mathbb{E}_X[\\log(1+fX \quad \mid \quad p=t)]\mathbb{P}_p(t)dt

""")

st.markdown("""
where $\mathbb{P}_p(t)$ is the density function for the beta distribution. 
""")

st.latex("""= \int_{0}^{1} (t\log(1+f) + (1-t)\log(1-f))\mathbb{P}_p(t)dt
""")

st.latex("""
= \mathbb{E}_p(p)\log(1+f) + \mathbb{E}_p(1-p)\log(1-f)
""")
st.latex("""
= \\frac{a+1}{a+b+2}\log(1+f) + \\frac{b+1}{a+b+2}\log(1-f)
""")

st.markdown("""
But by our work above, we already know that this is maximized when $f^* = (2a+2)/(a+b+2)-1$ $= (a-b)/(a+b+2)$. So, when $p\sim\\text{Beta}(18,4)$ we get that $f^* = 14/22 \\approx 64\%$. Below, you can see how $f^*$ varies as a function of the number of heads $a$ and tails $b$ that we observe. Notice too that if we did the "naive" thing of simply assuming that the probability of heads was $p = a/(a+b)$, we'd get the wrong answer for $f^*$ and thus leave money on the table.
""")

a_demo = st.number_input("Number of observed heads:", min_value=1, value=17)
b_demo = st.number_input("Number of observed tails:", min_value=1, max_value=a_demo, value=3)

def plot_return_curves(a,b):
    t = np.linspace(0,1,1001)
    p_t = stats.beta(a+1,b+1).pdf(t)
    p_t = p_t/p_t.sum()
    fs = np.linspace(0.01,0.99,98).reshape(-1,1)
    log_ev = (t*p_t*np.log(1+fs) + (1-t)*p_t*np.log(1-fs)).sum(axis=1)

    naive_f = 2*(a/(a+b))-1
    naive_idx = np.abs(fs-naive_f).argmin()
    optimal_idx = np.argmax(log_ev)
    optimal_f = fs[optimal_idx][0]

    plt.plot(fs,log_ev, color='black', label="True rate curve")
    naive_p = a/(a+b)
    plt.plot(fs, naive_p*np.log(1+fs) + (1-naive_p)*np.log(1-fs), color='black', linestyle='--', label="Naive rate curve")

    plt.plot(naive_f, log_ev[naive_idx], 'ro', color='tab:orange', label=f'Naive choice of f={100*naive_f:0.0f}% $\Rightarrow$ growth rate of {100*log_ev[naive_idx]:0.2f}%')
    plt.plot(optimal_f, log_ev[optimal_idx], 'ro', color='tab:blue', label=f'Optimal choice of f={100*optimal_f:0.0f}% $\Rightarrow$ growth_rate of {100*log_ev[optimal_idx]:0.2f}%')
    plt.legend()
    plt.plot(naive_f, naive_p*np.log(1+naive_f) + (1-naive_p)*np.log(1-naive_f), marker='o', color='tab:orange', fillstyle='none', label=f'Naive choice of f={100*naive_f:0.0f}% $\Rightarrow$ growth rate of {100*log_ev[naive_idx]:0.2f}%')
    plt.vlines([naive_f], log_ev[naive_idx], naive_p*np.log(1+naive_f) + (1-naive_p)*np.log(1-naive_f), linestyle=':', color='tab:orange')
    plt.title("Comparison of Optimal vs Naive $f^*$")

    st.pyplot()
plot_return_curves(a_demo,b_demo)

st.markdown("""
One way of interpreting what's happening here is that the Bayesian $f^*$ (blue) is optimizing for the true return curve (solid black), while the "naive" $f^*$ (orange) is optimizing for the naive curve (dashed black). You can see how this would work if you have an *informative* prior instead. If your prior is that $p\sim\\text{Beta}(a',b')$ and then you flip the coin $n$ times and observe $a$ heads and $b=n-a$ tails, your posterior is $p\sim\\text{Beta}(a+a',b+b')$. Somebody who doesn't have this prior would be optimizing $f$ for $\\text{Beta}(a,b)$, but you're optimizing optimizing for $\\text{Beta}(a+a',b+b')$.
""")

st.header("Beating the Market")

st.markdown("""

Now let's put these ideas together. There's a marketplace for coin flips. The market maker mints a coin, which has some unknown (and uniformly distributed) probability $p$ of landing heads. Market participants can go long or short the next coin flip. If A is long and B is short, and the coin comes up heads, then B pays A \$100. The price of a contract is set by the market. So, say, the price is \$90. That means that A paid \$90 for his/her long contract, and when B went short, he/she got paid \$90. So, in this example, A risked \$90 and made \$10, and B risked \$10 and lost \$10. This is close to how prediction markets like PredictIt work.

Let's say that the market has observed $n$ coin flips, $a$ of which were heads and $b$ of which were tails. Then the market "thinks" that $p \sim \\text{Beta}(a+1, b+1)$ and "believes" that the next flip $X_{n+1}$ is is a Bernoulli trial with probability $p$ of heads. The price of going long will be $p \cdot \$100$ and the price of going short will be $(1 - p) \cdot \$100$.

But now let's imagine that we know something the market does not. Let's say that we get the information equivalent of *five* extra flips of the coin. You can imagine this as inside information (we get to see some secret coin flips) or as better research (we built a 3D scanner and a physics engine that let us simulate coin flips). Either way, the rest of the market is left with a $\\text{Beta}(a+1,b+1)$ distribution, but we have a have a tighter $\\text{Beta}(a+a'+1, b+b'+1)$ distribution (where $a' + b' = 5$).

How much is this worth? Well, from our work in the **Bayesian Inference** section above, we know that we'll be acting as if $p=(a+a'+1)/(a+a'+b+b'+2)$, and the market will be acting as if $p=(a+1)/(a+b+2)$.


Let's write $\hat{p} = (a+1)/(a+b+2)$ for the market's point estimate of $p$, and let $\\tilde{p} = (1 + a + a')/(2 + a + b + a' + b')$ be our point estimate of $p$. If $\\tilde{p} > \hat{p}$, then from our earlier work in the **Betting with an Edge** section, we know that we're going to bet $(\\tilde{p}/\hat{p}-1)/(1/\hat{p}-1)$ of our wealth and that this is going to give us an edge of $D_{KL}(\\tilde{p} || \hat{p})$. Below, you can see what this looks like for different values of $a$ (the number of heads we "secretly" observe) and $a'$ (the number of heads observed by all market participants).

""")


a_prime = st.number_input("Number of Heads we Secretly Observe:", min_value=0, max_value=5, value=5)
b_prime = 5 - a_prime
a = st.number_input("Number of Heads Observed by All:", min_value=0, max_value=5, value=3)
b = 5 - a

def plot_final_example(a, a_prime, b, b_prime):
    our_p = (1+a+a_prime)/12
    market_p = (1 + a)/7

    x = np.linspace(0,1,101)
    pd.DataFrame({
        "our_distribution": stats.beta(1+a_prime+a, 1+b_prime+b).pdf(x),
        "market_distribution": stats.beta(1+a, 1+b).pdf(x)
    }, index=x).plot()
    plt.vlines([our_p], 0, stats.beta(1+a_prime+a, 1+b_prime+b).pdf(our_p), color='tab:blue', linestyle=':')
    plt.vlines([market_p], 0, stats.beta(1+a, 1+b).pdf(market_p), color='tab:orange', linestyle=':')
    plt.annotate("$\\tilde{p}=$" + f"${our_p:0.2f}$", xy=(our_p+0.02, 0.1), color='tab:blue')
    plt.annotate("$\hat{p}=$" + f"${market_p:0.2f}$", xy=(market_p+0.02, 0.3), color='tab:orange')
    plt.ylabel("Density")
    plt.xlabel("Possible Values for $p$")
    plt.title("Posterior Distributions for $p$\nOurs vs the Market")

    st.pyplot()

    fs = np.linspace(0,1,101)

    # Go long if our_p > market_p, else go short
    if our_p >= market_p:
        our_curve = our_p*np.log(1-fs + fs*(1.0/market_p)) + (1-our_p)*np.log(1-fs)
        f_star = (our_p/market_p-1)/(1/market_p - 1)
    else:
        our_curve = our_p*np.log(1-fs) + (1-our_p)*np.log(1-fs + fs*(1.0/(1-market_p)))
        f_star = ((1-our_p)/(1-market_p)-1)/(1/(1-market_p) - 1)
    market_curve = market_p*np.log(1-fs + fs*(1.0/market_p)) + (1-market_p)*np.log(1-fs)

    edge = our_p*np.log(our_p/market_p) + (1-our_p)*np.log((1-our_p)/(1-market_p))

    x = np.linspace(0,1,101)
    pd.DataFrame({
        "our_curve": our_curve,
        "market_curve": market_curve,
        "zero_return_line": [0]*len(fs)
    }, index=fs).plot(color=['black', 'black', 'tab:red'], style=['-','--', '--'])
    plt.plot(f_star, edge, 'ro', color='tab:blue')
    plt.vlines([f_star], -1, 0.25, color='tab:blue', linestyle=':')
    plt.hlines([edge], 0, 1, color='tab:blue', linestyle=':')
    plt.xlim(0,1)
    plt.ylim(-1,0.25)
    plt.annotate("$x=(\\tilde{p}/\hat{p}-1)/(1/\hat{p}-1)$", xy=(f_star+0.02, -0.97), color='tab:blue')
    plt.annotate("$y=D(\\tilde{p}||\hat{p})$", xy=(0.02, edge+0.02), color='tab:blue')
    plt.legend()
    plt.ylabel("Exponential Rate of Return")
    plt.xlabel("Value of $f$")
    plt.title("Return Curves for Choices of $f$\nOurs vs the Market")
    st.pyplot()

plot_final_example(a, a_prime, b, b_prime)
