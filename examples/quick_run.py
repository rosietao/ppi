import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from ppi_py.experiments import run
import pandas as pd

ns = [50, 100, 500]
ps = [2, 10, 20, 50, 100, 200, 300, 350, 400, 500, 600, 800, 1000, 1400, 1450, 1500, 2000, 3000]
df = run(
    ns=ns, ps=ps,
    N = 1500, num_trials=30,
    noise=1.0, sig=2.0, bet=0.0
)

df.to_csv("results/output_all.csv", index=False)