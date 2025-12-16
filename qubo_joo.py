#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports - General purposes
import itertools
import random
import numpy as np
from itertools import combinations
import csv
import pandas as pd

# Imports - D-Wave
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import ConstrainedQuadraticModel, BinaryQuadraticModel
from dimod.binary_quadratic_model import BinaryQuadraticModel
from itertools import combinations
from neal import SimulatedAnnealingSampler
from dimod.serialization.format import Formatter
import dwave.inspector


# In[2]:


# Imports - Qiskit
from docplex.mp.model import Model
from qiskit import Aer, IBMQ
from qiskit.algorithms.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver,VQE,SamplingVQE
from qiskit.algorithms.optimizers import SPSA, COBYLA,SLSQP
from qiskit.circuit.library import EfficientSU2
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import from_docplex_mp
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.utils import QuantumInstance
from qiskit.visualization import *
from qiskit.tools.visualization import circuit_drawer
from qiskit_optimization.runtime import VQEClient
from qiskit_optimization.runtime import QAOAClient
from qiskit.circuit.library import QAOAAnsatz
import matplotlib.pyplot as plt
from typing import List, Tuple
from qiskit.primitives import Sampler
from qiskit.circuit.library import TwoLocal
from qiskit import *


# In[3]:


# ================================================================================================= #
# ================================================================================================= #
#                                                                                                   #
# -------------------------------------JOIN ORDER OPTIMIZATION------------------------------------- #
#                                                                                                   #
# ================================================================================================= #
# ================================================================================================= #


# ================================================================================================= #
# ----------------------------------------QUBO Formulation----------------------------------------- #
# ================================================================================================= #
class QUBO_formulation:
    """
    The QUBO-formulation class is used to create a quadratic model with the aim of finding the optimal join order.
    """

    def relation_sublists(relations):
        # Creates the power set from the relations
        relations_sublists = sum([list(map(list, combinations(relations, i))) for i in range(len(relations) + 1)], [])

        # Removes the superfluous initial relations and the empty set from the power set
        for i in range(len(relations)+1):
            relations_sublists.pop(0)
        return relations_sublists


    def construct_QUBO(relations):
        """
        Method for creating the QUBO model.

        Parameter:
        relations: An n-element list of the relations involved in the join tree.
        """

        # Forming the power set and removing the single relations and the empty set
        relations_sublists = QUBO_formulation.relation_sublists(relations)

        # Number of relation combinations
        n = len(relations_sublists)

        # S_1
        s_1_helper = relations_sublists

        # S2_new
        s_2_helper = []
        for i in range(n):
            for j in range(i,n):
                if (i != j):
                    if(set(relations_sublists[i]).intersection(set(relations_sublists[j]))):
                        if(len(relations_sublists[i])<=len(relations_sublists[j])):
                            if(not set(relations_sublists[i]).issubset(set(relations_sublists[j]))):
                                s_2_helper.append([i,j])
                                            
        return s_1_helper, s_2_helper, n
# ================================================================================================= #


# In[4]:


# ================================================================================================= #
# -----------------------------------solution algorithms Qiskit------------------------------------ #
# ================================================================================================= #
class Solvers_qiskit:
    """
    The Solvers class is used to solve the modeled QUBO problem using Qiskit.
    """
    shots = 1000

    def prepare_model(s_1_helper, s_2_helper, n, weights):
        """
        Method for generating the QUBO problem.

        Parameter:
        s_1_helper: List of relation combinations
        s_2_helper: Index structure of the constraints S_2
        n: Number of relation combinations
        weights: list of weights
        """
        # Initialization of the QUBO model
        mdl = Model(name='Join Ordering')

        # Definition of the binary random variable
        p = [mdl.binary_var() for i in range(0, n)]

        w_max = max(weights) * 2

        # Setting up the Hamilton formula
        s_1 = mdl.sum(p[i] * (weights[i] - w_max) for i in range(0, len(s_1_helper)))
        s_2 = mdl.sum(w_max * p[s_2_helper[i][0]] * p[s_2_helper[i][1]] for i in range(0, len(s_2_helper)))
        objective = s_1 + s_2

        # Setting the goal of optimization
        mdl.minimize(objective)
        quadratic_program = from_docplex_mp(mdl)

        return quadratic_program

    # Calculation of the exact result
    def exact_result(quadratic_program):
        """
        Method for exactly calculating an optimal join order.

        Parameter:
        quadratic_program: Quadratic Programm, on which to run the optimization algorithm.
        """
        exact_mes = NumPyMinimumEigensolver()
        exact = MinimumEigenOptimizer(exact_mes)
        results_exact = exact.solve(quadratic_program)
        return results_exact
    

    # Simulation of the VQE algorithm with sampling
    def sampling_vqe_sim(quadratic_program, local_optimizer):
        """
        Method for simulating a quantum computer using the VQE algorithm.

        Parameter:
        quadratic_program: Quadratic Programm, on which to run the optimization algorithm.
        """
        iterations = 125
        if(local_optimizer == 'cobyla'):
            optimizer = COBYLA(maxiter=iterations)
        if(local_optimizer == 'spsa'):
            optimizer = SPSA(maxiter=iterations)
        operator, offset = QuadraticProgramToQubo().convert(quadratic_program).to_ising()
        ansatz = TwoLocal(operator.num_qubits,rotation_blocks="ry",entanglement_blocks="cz",reps=1, entanglement="linear")
        quantum_method_vqe = SamplingVQE(sampler=Sampler(),ansatz=ansatz,optimizer = optimizer)
        min_eigen_optimizer_vqe = MinimumEigenOptimizer(min_eigen_solver = quantum_method_vqe)
        results_vqe = min_eigen_optimizer_vqe.solve(quadratic_program)
        return results_vqe
    
    # Simulation of the VQE algorithm
    def vqe_sim(quadratic_program, local_optimizer):
        """
        Method for simulating a quantum computer using the VQE algorithm.

        Parameter:
        quadratic_program: Quadratic Programm, on which to run the optimization algorithm.
        """
        if(local_optimizer == 'cobyla'):
            optimizer = COBYLA(maxiter=125)
        if(local_optimizer == 'spsa'):
            optimizer = SPSA(maxiter=125)
        quantum_instance_vqe = QuantumInstance(backend = Aer.get_backend('qasm_simulator'), shots=Solvers_qiskit.shots)
        quantum_method_vqe = VQE(quantum_instance = quantum_instance_vqe, optimizer = optimizer)
        min_eigen_optimizer_vqe = MinimumEigenOptimizer(min_eigen_solver = quantum_method_vqe)
        results_vqe = min_eigen_optimizer_vqe.solve(quadratic_program)
        return results_vqe
    
    # Simulation of the QAOA algorithm
    def qaoa_sim(quadratic_program, local_optimizer):
        """
        Method for simulating a quantum computer using the QAOA algorithm.

        Parameter:
        quadratic_program: Quadratic Programm, on which to run the optimization algorithm.
        """
        if(local_optimizer == 'cobyla'):
            optimizer = COBYLA(maxiter=125)
        if(local_optimizer == 'spsa'):
            optimizer = SPSA(maxiter=125)
        if(local_optimizer=='slsqp'):
            optimizer = SLSQP(maxiter=125)
        #operator, offset = QuadraticProgramToQubo().convert(quadratic_program).to_ising()
        qaoa_mes = QAOA(sampler=Sampler(),reps=1,optimizer = optimizer)
        qaoa_meo = MinimumEigenOptimizer(min_eigen_solver = qaoa_mes)
        results_qaoa = qaoa_meo.solve(quadratic_program)
        return results_qaoa
    

    def callback(nfev, parameters, energy, stddev):
        """
        Helper method for returning values.

        Parameter:
        nfev: The number of executions.
        parameters: The parameters with which the algorithm is run in each iteration
        energy: The intermediate result of the objective function.
        stddev: The standard deviation.
        """
        intermediate_info = {
            'nfev': [],
            'parameters': [],
            'energy': [],
            'stddev': []
        }

        intermediate_info['nfev'].append(nfev)
        intermediate_info['parameters'].append(parameters)
        intermediate_info['energy'].append(energy)
        intermediate_info['stddev'].append(stddev)
# ================================================================================================= #


# In[5]:


# ================================================================================================= #
# -----------------------------------Solution Algorithms - D-Wave---------------------------------- #
# ================================================================================================= #
class Solvers_dwave:
    """
    The Solvers_dwave class is used to solve the modeled QUBO problem on D-Wave quantum computers.
    """
    def prepare_model(s_1_helper, s_2_helper, n, weights):
        """
        Method for generating the QUBO problem.

        Parameter:
        s_1_helper: List of relation combinations
        s_2_helper: index structure of the constraints S_2
        n: number of relation combinations
        weights: list of weights
        """
        w_max = 2 * max(weights)

        constrained_quadratic_model = ConstrainedQuadraticModel() # initialize the quadratic model.
        objective = BinaryQuadraticModel(vartype='BINARY') # initialize the objective.

        for i in range(n):
            objective.add_variable(i)

        # S_1
        for i in range(0, len(s_1_helper)):
            objective.set_linear(i, (weights[i] - w_max))
        
        # S_2
        for i in range(0, len(s_2_helper)):
            objective.set_quadratic(s_2_helper[i][0], s_2_helper[i][1], + w_max)
        

        constrained_quadratic_model.set_objective(objective)

        return constrained_quadratic_model
    
    def exact_result(constrained_quadratic_model):
        """
        Method for exactly calculating an optimal join order.

        Parameter:
        quadratic_program: Quadratic program on which to run the optimization algorithm.
        """
        binary_quadratic_model, invert = dimod.cqm_to_bqm(constrained_quadratic_model)
        result = dimod.ExactSolver().sample(binary_quadratic_model)
        return result

    def simulated_annealing(constrained_quadratic_model):
        """
        Method for calculating an optimal join order using simulated annealing.

        Parameter:
        quadratic_program: Quadratic program on which to run the optimization algorithm.
        """
        binary_quadratic_model, invert = dimod.cqm_to_bqm(constrained_quadratic_model)
        result = SimulatedAnnealingSampler().sample(binary_quadratic_model, num_reads=10000)
        return result
    
    def dwave_sampler(constrained_quadratic_model):
        """
        Method for calculating an optimal join order using D-Wave QPU.

        Parameter:
        quadratic_program: Quadratic program on which to run the optimization algorithm.
        """
        endpoint = 'https://cloud.dwavesys.com/sapi'
        token = 'token_here'
        #solver = 'DW_2000Q_6'
        solver = 'Advantage_system4.1'

        binary_quadratic_model, invert = dimod.cqm_to_bqm(constrained_quadratic_model)
        dw = DWaveSampler(endpoint=endpoint, token=token, solver=solver)
        sampler = EmbeddingComposite(dw)

        result = sampler.sample(binary_quadratic_model, num_reads=1000)

        return result

# ================================================================================================= #


# In[6]:


# ================================================================================================= #
# ------------------------------------class with helper methods------------------------------------- #
# ================================================================================================= #
class Helping_functions:
    """
    This class contains helper functions for the experimental execution of the QUBO algorithm.
    """
    def init_weigths(relations):
        """
        Helper method for initializing random join costs.

        Parameter:
        relations: A list of relations.
        """
        powerset = sum([list(map(list, combinations(relations, i))) for i in range(len(relations) + 1)], [])
        return [random.randint(1, 10) for _ in range(len(powerset)-(len(relations) + 1))]
    

    def print_relations_and_weights(relations, weights):
        """
        Auxiliary method for visualizing all possible relation combinations and their weights.

        Parameter:
        relations: A list of relations.
        weights: Cost of each join.
        """
        file = open("QUBO/Qiskit-Tests.txt", "a", encoding='utf-8')
        relation_combinations = QUBO_formulation.relation_sublists(relations)
        # Overview of all sets and their weights
        for i in range (len(relation_combinations)):
            print('Variable:-', i, ' (weight:-', weights[i], ') -> ', list(relation_combinations[i]),sep="")
            file.write('\n')
            file.write('Variable ')
            file.write(str(i))
            file.write(' (weight: ')
            file.write(str(weights[i]))
            file.write(') -> ')
            file.write(str(list(relation_combinations[i])))
        file.write('\n')
        file.close()

    def print_qubo_results_qiskit(quadratic_program, relations, result):
        """
        Helper method for outputting the results of a Qiskit experiment.

        Parameter:
        quadratic_program: Quadratic program on which to run the optimization algorithm.
        relations: A list of relations.
        result: Result of the optimization.
        """
        # file output
        file = open("Qiskit-Tests.txt", "a", encoding='utf-8')
        file.write('\n')
        file.write(str(quadratic_program))
        file.write('\n')
        file.write(str(result))
        file.write('\n')

        # solution set
        relations_sublists = QUBO_formulation.relation_sublists(relations)
        solution_set = []
        optimization_result = result.x.tolist()
        for x in range(len(optimization_result)):
            if optimization_result[x] == 1.0:
                solution_set.append(relations_sublists[x])
        file.write('\n')
        file.write(str(solution_set))
        file.write('\n')
        file.write('=====================================================================================')
        file.close()

        print('Used Joins: ')
        print(str(solution_set))

    def print_qubo_results_dwave(quadratic_model, relations, result):
        """
        Helper method to output the results of a D-Wave experiment.

        Parameter:
        quadratic_program: Quadratic program on which to run the optimization algorithm.
        relations: A list of relations.
        result: Result of the optimization.
        """
        relation_combinations = QUBO_formulation.relation_sublists(relations)
        joins_q = []
        for i in range(len(result.lowest().record[0][0])):
            if(result.lowest().record[0][0][i] == 1):
                joins_q.append(list(relation_combinations[i]))
        file = open("D-Wave-Tests.txt", "a", encoding='utf-8')
        file.write(str(result.lowest()))
        file.write('\n')
        file.write(str(joins_q))
        file.write('\n')
        file.write('=====================================================================================')
        file.write('\n')
        file.write('\n')
        file.close()
    
    def show_number_of_qubits_dwave(result):
        """
        Helper method to display the number of qubits used in embedding.

        Parameter:
        result: Result of the calculation of the D-Wave quantum computer.
        """
        embedding = result.info['embedding_context']['embedding']
        print(f"Number of logical variables: {len(embedding.keys())}")
        print(f"Number of physical qubits used in embedding: {sum(len(chain) for chain in embedding.values())}")
        
    def find_circuit_depth_qiskit(relations,weights,solver):
        """
        To find the depth of the simulator either by QAOA or VQE method.
        
        Parameter:
        relations: input relations
        solver: The solver being used to solve the problem.
        """
        # Generation and calculation of the QUBO:
        a,b,c,d,e = QUBO_formulation.construct_QUBO(relations)
        quadratic_program = Solvers_qiskit.prepare_model(a, b, c, d, e, weights)
        operator, offset = QuadraticProgramToQubo().convert(quadratic_program).to_ising()

        if (solver=='vqe_sim'):
            ansatz = TwoLocal(operator.num_qubits,rotation_blocks="ry", entanglement_blocks="cz")
            parameter=np.random.random_sample(ansatz.num_parameters)
            ckt=VQE().construct_circuit(parameter=parameter,operator=operator)
            ckt_depth=ckt[0].decompose().depth()
        if (solver=='qaoa_sim'):
            ansatz=QAOAAnsatz(operator).decompose()
            num_parameters=ansatz.num_parameters
            parameters=np.random.uniform(low=0, high=1, size=(num_parameters,))
            ckt=QAOA().construct_circuit(parameter=parameters,operator=operator)
            ckt=ckt[0].decompose().decompose()
            ckt_depth=ckt.depth()
        return ckt_depth

    def subsetsFromBinaryString(values, relations):
        relation_combinations = QUBO_formulation.relation_sublists(relations)
        result=list()
        for i in range(len(values)):
            if values[i]==1:
                result.append(relation_combinations[i])
        return result


    def quboToTree(result, relations):
        relation_combinations = QUBO_formulation.relation_sublists(relations)
        vars=list()
        for i in range(len(result)):
            if result[i]==1:
                vars.append(relation_combinations[i])
        if len(vars)!=len(relations)-1:
            return [], vars, "Wrong number of joins. Should be {} but {} found.".format(len(relations)-1,len(vars))
        forest = {}
        for r in relations:
            forest[frozenset([r])] = [r]
        for v in vars:
            for [left, right] in itertools.combinations(forest,2):
                # left and right have to be disjoint and together form the set v
                if left.isdisjoint(right) and left.union(right)==set(v):
                    # remove single element list [a]+[b]==[a,b]!=[[a],[b]]
                    leftValue = forest[left][0] if len(left)==1 else forest[left]
                    rightValue = forest[right][0] if len(right)==1 else forest[right]
                    # add new tree to forest
                    forest[frozenset(v)]= [leftValue, rightValue]
                    # remove used trees
                    forest.pop(left)
                    forest.pop(right)
        if len(forest)==1:
            return forest.popitem()[1], vars, None
        else: 
            return [] , vars, "More than one tree remained at the end"

    def dynamic_programming(relations, cost):
        t = {}
        relations_sublists = QUBO_formulation.relation_sublists(relations)
        sets = [frozenset([e]) for e in relations]
        sets += [frozenset(e) for e in relations_sublists]
        for e in relations:
            t[frozenset([e])] = (e,0)
        n=len(relations)
        for i, S in enumerate(relations_sublists):
            #if(len(S)==s):
            #while(len(S)==s):
                values = []
                for (a,b) in itertools.combinations(sets,2):
                    if a.isdisjoint(b) and a.union(b)==frozenset(S):
                        values.append(([t[a][0],t[b][0]], t[a][1] + t[b][1] + cost[i]))
                t[frozenset(S)] = (min(values, key=lambda e: e[1]))
        #s+=1
        return t[frozenset(relations)]

# ================================================================================================= #


# ================================================================================================= #
# -----------------------------------Base class for execution-------------------------------------- #
# ================================================================================================= #
class JooQuboSolver:

    def __init__(self, filename):
        self.logfile = open("servers/qubo/{}.log".format(filename),"w")

    def __del__(self):
        self.logfile.close()

    def logQiskitResult(self, result, relations, weights):
        self.logfile.write("# {} ; {}\n".format(",".join(relations), ",".join(str(e) for e in weights)))
        for r in result.aggregate().data():
            logData = [",".join(relations)]
            logData.append(r.num_occurrences)
            logData.append(r.energy)
            order, variables, error=Helping_functions.quboToTree(r.sample, relations)
            cost =0
            for i in range(len(r.sample)):
                if r.sample[i]==1:
                    cost+=weights[i]
            logData.append("Valid" if error==None else "Invalid")
            logData.append(order)
            logData.append(variables)
            logData.append(cost)
            self.logfile.write(";".join([str(e) for e in logData]))
            self.logfile.write("\n")
            self.logfile.flush()

    """
    This class is used to run the algorithm with a defined set of relations and weights
    """
    def qiskit_experiment(self, relations, weights,solver, local_optimizer):
        """
        Method for performing measurements in the Qiskit framework.

        Parameter:
        relations: A list of input relations.
        weights: Weights of all possible joins
        solver: The solver to use
        """

        # Generation and calculation of the QUBO:
        a, b, c = QUBO_formulation.construct_QUBO(relations)
        quadratic_program = Solvers_qiskit.prepare_model(a, b, c, weights)

        #Helping_functions.print_relations_and_weights(relations, weights)

        if(solver == 'exact_result'):
            result = Solvers_qiskit.exact_result(quadratic_program, local_optimizer).x
        if(solver == 'qaoa_sim'):
            result = Solvers_qiskit.qaoa_sim(quadratic_program, local_optimizer,).x
        if(solver == 'vqe_sim'):
            result = Solvers_qiskit.vqe_sim(quadratic_program, local_optimizer).x
        if(solver == 'sampling_vqe_sim'):
            result= Solvers_qiskit.sampling_vqe_sim(quadratic_program, local_optimizer).x
        return Helping_functions.quboToTree(result, relations)

    def logDwaveResult(self, result: dimod.SampleSet, relations, weights,dp_optimal_cost,total_cost):
        self.logfile.write("# {} ; {}\n".format(",".join(relations), ",".join(str(e) for e in weights)))
        lowest_energy_state = result.aggregate().lowest().record[0][1]
        for r in result.aggregate().data():
            logData = [",".join(relations)]
            logData.append(r.num_occurrences)
            logData.append(r.energy)
            order, variables, error = Helping_functions.quboToTree(r.sample, relations)
            logData.append("Valid" if error==None else "Invalid")
            logData.append("Optimal" if total_cost == dp_optimal_cost and r.energy == lowest_energy_state else "Not Optimal")
            logData.append(order)
            logData.append(variables)
            self.logfile.write(";".join([str(e) for e in logData]))
            self.logfile.write("\n")
            self.logfile.flush()

    def dwave_experiment(self, relations, weights, solver):
        """
        Method of making measurements in the D-Wave framework.

        Parameter:
        relations: A list of input relations.
        weights: Weights of all possible joins
        solver: The solver to use
        """
        a,b,c = QUBO_formulation.construct_QUBO(relations)
        #new_weights_list = Helping_functions.get_new_weights_list(relations, weights)
        constrained_quadratic_model = Solvers_dwave.prepare_model(a, b, c, weights)
        t = Helping_functions.dynamic_programming(relations, weights)
        dp_optimal_cost = t[1]

        #Helping_functions.print_relations_and_weights(relations, weights)

        if(solver == 'exact_result'):
            result = Solvers_dwave.exact_result(constrained_quadratic_model)
        if(solver == 'dwave'):
            result = Solvers_dwave.dwave_sampler(constrained_quadratic_model)
            #Helping_functions.show_number_of_qubits_dwave(result)
        if(solver == 'simulated_annealing'):
            result = Solvers_dwave.simulated_annealing(constrained_quadratic_model)

        lowest_energy_state = result.lowest().record[0][0]
        total_cost = 0
        for i in range(len(lowest_energy_state)):
            if lowest_energy_state[i] == 1:
                total_cost += weights[i]
        self.logDwaveResult(result, relations, weights, dp_optimal_cost, total_cost)
        return Helping_functions.quboToTree(result.lowest().record[0][0], relations)
