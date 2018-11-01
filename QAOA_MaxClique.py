import qiskit
import numpy as np
import matplotlib.pyplot as plt
import json
from graph import *

# Random comment
P =1
def makeCircuit(inbits, outbits):
    q = qiskit.QuantumRegister(inbits+outbits)
    c = qiskit.ClassicalRegister(inbits+outbits)
    qc = qiskit.QuantumCircuit(q, c)

    q_input = [q[i] for i in range(outbits,outbits+inbits)] 
    q_output = [q[j] for j in range(outbits)]

    return qc, c, q_input, q_output

# measure all qubits in q_input register, return dictionary of samples
def measureInput(qc, q_input, c):
    for i in range(len(q_input)):
        qc.measure(q_input[i], c[i])
    job = qiskit.execute(qc, backend='local_qasm_simulator', shots=1024)
    return job.result().get_counts(qc)

def test5(qc, q_input, c):
    data = measureInput(qc, q_input, c)
    # assemble data from dictionary into list
    parsed = []
    xticks = []
    n = len(q_input)
    for i in range(2**n):
        bits = np.binary_repr(i, width=n)
        xticks.append(bits)
        bits += "00"
        if bits in data: parsed.append(data[bits])
        else: parsed.append(0)
    plt.bar(range(2**n), parsed)
    plt.xticks(range(2**n),xticks,rotation="vertical")
    plt.xlabel('Outcomes')
    plt.ylabel('Counts')
    plt.title('Measurement Histogram')
    plt.show()

def applyQAOA(gamma, beta, graph):
    ### INIT REGS
    qc, c, q_input, q_output = makeCircuit(graph.getNumNodes(), 1);
    PENALTY = graph.getMaxEdges()
    ### H on every input register
    for node in q_input:
        qc.h(node)
    complement = graph.getEdgesComp();
    edges = graph.getEdges()
    ### APPLY V AND W
    for i in range(P):
        ### APPLY V
        # EDGES IN THE GRAPH
        for edge in edges:
            nodeList = edge.getNodes()
            qc.cu1(-gamma, q_input[nodeList[0].name], q_input[nodeList[1].name])
        # EDGES NOT IN THE GRAPH
        for edge in complement:
            nodeList = edge.getNodes()
            qc.cu1(PENALTY*gamma, q_input[nodeList[0].name], q_input[nodeList[1].name])
            
        ### APPLY W
        for node in q_input:
            qc.h(node)
            qc.u1(2*beta, node)
            qc.h(node)

    ### Measure
    results = measureInput(qc, q_input, c)
    ### Compute the result expectation
    

    ### Parse the result list.
    # B/c we only care about counts associated with input register
    # we combine the counts of states with same input register bits
    
    counts = dict()
    for key in results:
        if key[1:] not in counts:
            counts[key[1:]] = results[key]
        else:
            counts[key[1:]] += results[key] 

    
    #print(counts)
    eox = 0
    eox2 = 0

    for val in counts:
        cliqNum = 0
        for edge in edges:
            nodeList = edge.getNodes()
            #print("Node 1:", nodeList[0].name,"Node 2:", nodeList[1].name)
            if val[nodeList[0].name] == '1' and val[nodeList[1].name] == '1':
                cliqNum += 1
        for edge in complement:
            nodeList = edge.getNodes()
            if val[nodeList[0].name] == '1' and val[nodeList[1].name] == '1':
                cliqNum -= PENALTY
        eox += counts[val]/1024 * cliqNum
        eox2 += (cliqNum**2) * counts[val]/1024
    std = np.sqrt((len(counts)/(len(counts) -1))*(eox2 - eox**2))
    return eox, std

### gradient ascent optimizer
# graph is graph to optimize over
# epsilon controls how far out the delta is calculated
# eta is learning rate
# threshold is the average of gamma and beta that we will consider a max

def optimize(graph, epsilon, eta, threshold):
    count = 0
    gamma = 2
    beta = 2
    dgamma = (applyQAOA(gamma + epsilon, beta, graph) - applyQAOA(gamma - epsilon, beta, graph))/(2*epsilon)
    dbeta = (applyQAOA(gamma, beta + epsilon, graph) - applyQAOA(gamma, beta + epsilon, graph))/(2*epsilon)
    flipper = True #Alternate between maxing gamma and maxing beta
    while((abs(dgamma) + abs(dbeta))/2 > threshold):
        if(flipper):
            if (dgamma > 0): 
                gamma = (gamma + (dgamma * eta)) % (2*np.pi)
            elif (dgamma < 0):
                gamma = (gamma - (dgamma * eta)) % (2*np.pi)
            dgamma = (applyQAOA(gamma + epsilon, beta, graph) - applyQAOA(gamma - epsilon, beta, graph))/(2*epsilon)
        else:
            if(dbeta > 0):
                beta = (beta + (dbeta * eta)) % np.pi
            elif (dbeta < 0):
                beta = (beta - (dbeta * eta)) % np.pi
            dbeta = (applyQAOA(gamma, beta + epsilon, graph) - applyQAOA(gamma, beta + epsilon, graph))/(2*epsilon)
            
        count+=1
        print("Count", count, "dg", dgamma, "db", dbeta)
        flipper = not flipper
    
    print(count)
    return gamma, beta

def main():
    
    ###TESTING GRAPH
    #0---1
    #| / |
    #3---2
    myGraph = Graph(0, 0)
    nodes = [Node(i) for i in range(4)]

    edges = []
    edges.append(Edge(nodes[0], nodes[1]))
    edges.append(Edge(nodes[1], nodes[2]))
    edges.append(Edge(nodes[2], nodes[3]))
    edges.append(Edge(nodes[3], nodes[0]))
    edges.append(Edge(nodes[3], nodes[1]))

    for n in nodes:
        myGraph.addNode(n)
    
    for e in edges:
        myGraph.addEdge(e)

        
    ### Run the algorithm
    #expect = applyQAOA(gamma, beta, myGraph)
    #print("Expectation Value:", expect)

    ### OPTIMIZE

    #bestGamma, bestBeta = optimize(myGraph, 0.1, 0.1, 0.05)
    #print("BestGamma: ", bestGamma, "bestBeta", bestBeta)
    #print("Optimized Expectation value", applyQAOA(bestGamma, bestBeta, myGraph))
    #print("Optimal Gamma:", bestGamma, "Optimal Beta:", bestBeta)
    #BestGamma:  4.6015625 bestBeta 0.18702062766020688
    #Optimized Expectation value -0.3115234375

    ### Make graphs.
    # I'm thinking we hold one variable constant at its maxed value
    # and vary the other and vice versa.
    # Gamma has a larger range than beta. Do we want more data points for gamma than beta?
    # The last page of the worksheet says exactly which graphs we need in our report
    # so make sure we have at least those

    BestGamma = 4.6015625
    BestBeta = 0.18702062766020688
    betas = np.linspace(0, np.pi, 10)
    gammas = np.linspace(0, 2*np.pi, 100)
    varyingBeta = []
    varyingGamma = []
    betaSTD = []
    gammaSTD = []
    
    y = []
    std = []
    
    for gammaa in gammas:
        e, s = applyQAOA(gammaa, BestBeta, myGraph)
        y.append(e)
        std.append(s)

    with open("varyingGamma.txt", 'w') as f:
        json.dump(y, f)

    with open("gammaSTD.txt", 'w') as f:
        json.dump(std, f)
    """
    y = []
    std = []
    for betaa in betas:
        e, s = applyQAOA(BestGamma, betaa, myGraph)
        y.append(e)
        std.append(s)
        
    with open("varyingBeta.txt", 'w') as f:
        json.dump(y, f)

    with open("betaSTD.txt", 'w') as f:
        json.dump(std, f)
    """   
    with open("varyingGamma.txt", 'r') as f:
        varyingGamma = json.load(f)
    
    #with open("varyingBeta.txt", 'r') as f:
    #   varyingBeta = json.load(f)

    #with open("betaSTD.txt", 'r') as f:
    #    betaSTD = json.load(f)

    with open("gammaSTD.txt", 'r') as f:
        gammaSTD = json.load(f)
    
    #betaG = plt.errorbar(betas, varyingBeta, betaSTD, ecolor='black', elinewidth = 0.5, capsize=3)
    gammaG = plt.errorbar(gammas, varyingGamma, gammaSTD, ecolor='black', elinewidth = 0.5, capsize=3)
    plt.legend(('Gamma Graph',))
    plt.xlabel('Gamma values')
    plt.ylabel('Expectation Value')
    plt.title('Expectation Value vs Gamma holding Beta constant')
    plt.show()

main()
