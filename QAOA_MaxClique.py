import qiskit
import numpy as np
import matplotlib.pyplot as plt
from graph import *

PENALTY = 2;
P = 2;

def makeCircuit(x, y):
    q = qiskit.QuantumRegister(x+y)
    c = qiskit.ClassicalRegister(x+y)
    qc = qiskit.QuantumCircuit(q, c)

    q_input = [q[i] for i in range(y,x)]  # grover registers
    q_output = [q[j] for j in range(y)]                   # ancilla initialized to |->

    return qc, c, q_input, q_output

# measure all qubits in q_input register, return dictionary of samples
def measureInput(qc, q_input, c):
    for i in range(len(q_input)):
        qc.measure(q_input[i], c[i])
    job = qiskit.execute(qc, backend='local_qasm_simulator', shots=1024)
    return job.result().get_counts(qc)

def positiveConstraint(qc, q_i, q_j, gamma):
    qc.cu1(-gamma, q_i, q_j)
    return 0

def negativeConstraint(qc, q_i, q_j, gamma):
    qc.cu1(PENALTY*gamma, q_i, q_j)
    return 0

def test5(qc, q_input, c):
    data = measureInput(qc, q_input, c)
    # assemble data from dictionary into list
    parsed = []
    xticks = []
    n = len(q_input)
    for i in range(2**n):
        bits = np.binary_repr(i, width=n)
        xticks.append(bits)
        if bits in data: parsed.append(data[bits])
        else: parsed.append(0)

    plt.bar(range(2**n), parsed)
    plt.xticks(range(2**n),xticks,rotation="vertical")
    plt.xlabel('Outcomes')
    plt.ylabel('Counts')
    plt.title('Measurement Histogram')
    plt.show()

def main():
    gamma = []
    beta = []
    nodes = 3
    qc, c, q_input, q_output = makeCircuit(nodes+1,1);

    for i in range(P):
        gamma.append(np.random.uniform(0,2*np.pi))
    for i in range(P):
        beta.append(np.random.uniform(0,np.pi))
    
    myGraph = Graph(nodes, 0)
    complement = myGraph.getEdgesComp());
    edges = myGraph.getEdges()
    ### H on every input register
    for node in q_input:
        qc.h(node)

    ### APPLY V AND W
    for i in range(P):
        ### APPLY V
        # EDGES IN THE GRAPH
        for edge in edges:
            qc.cu1(-gamma, q_input[edge.node1], q_input[edge.node2])
        # EDGES NOT IN THE GRAPH
        for edge in complement:
            qc.cu1(gamma*PENALTY, q_input[edge.node1], q_input[edge.node2])

        ### APPLY W
        for node in q_input:
            qc.h(node)
            qc.u1(2*beta[i], node)
            qc.h(node)

    ### Measure. Then we need to optimize beta and gamma choices.
        
main()
