"""Step-0 PersistBench (GPU): the untrained policy on the PersistBench subset (see evaluation/)."""

from evaluation import evaluate


def run():
    return evaluate.run("step0")
