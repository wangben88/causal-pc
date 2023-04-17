import bnlearn
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import CausalInference



model = bnlearn.import_DAG('src/Experiments/bayesian_network/child.bif')
infer = CausalInference(model["model"])

print("Ground truth causal effect: ", infer.query(variables=["Sick"], do={
    "XrayReport": 0
}))

vars = bnlearn.topological_sort(model)

print("Topological order of variables: ", vars)


df = bnlearn.sampling(model, n=1000, methodtype='bayes')

# make variables binary
df = df.replace(2, 1)
df = df.replace(3, 1)
df = df.replace(4, 1)
df = df.replace(5, 1)
df = df.replace(6, 1)

print(df)




df.to_pickle("src/Experiments/data/child.pkl")

