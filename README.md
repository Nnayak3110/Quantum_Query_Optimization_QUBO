# Query Optimisation using Quantum Algorithms for DBMS by formulating it as QUBO

https://doi.org/10.1145/3579142.3594298

This repository contains quantum and classical algorithms to solve the join ordering problem in DBMS. 
It includes an experimental dataset, method implementation, and solving join ordering using both classical heuristics and quantum variational algorithms.

## 📌 Overview

Join order optimization is a core challenge in database management systems (DBMS), especially when considering bushy join trees, which significantly expand the search space compared to left-deep trees.

This work formulates the bushy join tree optimization problem as a Quadratic Unconstrained Binary Optimization (QUBO) problem and explores solving it using:

- Quantum algorithms (QAOA, VQE) on gate-based quantum simulators
- Execution on real quantum hardware using Quantum Annealing
- Classical baseline approaches for comparison, like Dynamic Programming

The goal is to demonstrate how quantum algorithms such as QAOA, VQE and Quantum Annealing can be applied to a fundamental DBMS optimization problem.

---



Citation:

@inproceedings{nayak2023constructing,
  title={Constructing optimal bushy join trees by solving qubo problems on quantum hardware and simulators},
  author={Nayak, Nitin and Rehfeld, Jan and Winker, Tobias and Warnke, Benjamin and {\c{C}}alikyilmaz, Umut and Groppe, Sven},
  booktitle={Proceedings of the international workshop on big data in emergent distributed environments},
  pages={1--7},
  year={2023}
}
