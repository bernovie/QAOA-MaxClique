import qiskit
import numpy as np
import matplotlib.pyplot as plt
import json
from graph import *
from matplotlib import cm
from qiskit.backends.aer import QasmSimulatorPy
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd

P = 1;

def isClique(graph, binString):
    nodes = graph.getNodes()
    edges = graph.getEdges()
    nodesRefined = []
    for i in range(len(nodes)):
        if binString[i] == '1':
            nodesRefined.append(i)
    
    for i in range(len(nodesRefined)):
        for j in range(i, len(nodesRefined)):
            tempE = Edge(nodes[nodesRefined[i]], nodes[nodesRefined[j]])
            if (tempE not in edges):
                return False
    return True

def classicalMC(graph):
    currentMax = 0
    maxedState = 0
    for i in range(2**graph.getNumNodes()):
        tempB = np.binary_repr(i, width = graph.getNumNodes())
        if(list(tempB).count('1') > currentMax):
            if(isClique(graph,tempB)):
                currentMax = list(tempB).count('1')
                maxedState = tempB
    return currentMax, maxedState

def getCost(maxedState, graph):
    edges = graph.getEdges()
    complement = graph.getEdgesComp()
    PENALTY = graph.getMaxEdges()
    cliqNum = 0
    for edge in edges:
        nodelist = edge.getNodes()
        if maxedState[nodelist[0].name] == '1' and maxedState[nodelist[1].name] == '1':
            cliqNum += 1
    for edgeComp in complement:
        nodelist = edgeComp.getNodes()
        if maxedState[nodelist[0].name] == '1' and maxedState[nodelist[1].name] == '1':
            cliqNum -= PENALTY
    return cliqNum

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
    job = qiskit.execute(qc, QasmSimulatorPy(), shots=1024)
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

# Gamma: 4.6015625 Beta: 0.18702062766020688
all_average_costs = []
all_minimum_costs = []
all_maximum_costs = []
all_graphs_average_cost = dict()
fbest = []
all_counts = []

myGraphTest = Graph(0,0)
nodes = [Node(i) for i in range(4)]
edges = []
edges.append(Edge(nodes[0], nodes[1]))
edges.append(Edge(nodes[1], nodes[2]))
edges.append(Edge(nodes[2], nodes[3]))
edges.append(Edge(nodes[3], nodes[0]))
edges.append(Edge(nodes[3], nodes[1]))

for n in nodes:
    myGraphTest.addNode(n)
    
for e in edges:
    myGraphTest.addEdge(e)

def applyQAOA(params,Graph=myGraphTest, showOutput=False):
    gamma, beta = params
    ### INIT REGS
    qc, c, q_input, q_output = makeCircuit(Graph.getNumNodes(), 1);
    PENALTY = Graph.getMaxEdges()
    ### H on every input register
    for node in q_input:
        qc.h(node)
    complement = Graph.getEdgesComp();
    edges = Graph.getEdges()
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

    sortedCounts = []
    for key in sorted(counts):
        sortedCounts.append((key, counts[key]))

    expectation = 0
    costs = []
    values = []
    min_cliqNum = 0
    max_cliqNum = 0
    values_at_max_cliqNum = 0
    dictionary_of_cost = dict()

    for val, sortedCount in sortedCounts:
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
        
        if (cliqNum < min_cliqNum):
            min_cliqNum = cliqNum
        
        if (cliqNum > max_cliqNum):
            max_cliqNum = cliqNum
            val_at_max_cliqNum = val
        
        expectation += sortedCount/1024 * cliqNum
        costs.append(cliqNum)
        values.append(val)
        dictionary_of_cost[val] = cliqNum
    max_vals = []
    
    for val in values:
        if dictionary_of_cost[val] == dictionary_of_cost[val_at_max_cliqNum]:
            max_vals.append(val)

    average_cost = sum(costs)/len(costs)
    all_average_costs.append(average_cost)
    all_maximum_costs.append(max_cliqNum)
    all_minimum_costs.append(min_cliqNum)
    currentMax,maxedState = classicalMC(Graph)
    currentMaxCost = getCost(maxedState, Graph)
    fbest = [cost/currentMaxCost for cost in costs]
    all_counts = [counts[val] for val in values]

    #density = graph.getNumNodes()/graph.getNumEdges()
    #if density not in all_graphs_average_cost:
    #    all_graphs_average_cost[density] = all_average_costs

    #print("The maximum cliques of these graph are: " + str(max_vals) + "\n where a 1 represents in the clique and a 0 not in the clique")
    
    if showOutput:
        colors = ["green" if dictionary_of_cost[value] == dictionary_of_cost[val_at_max_cliqNum] else "blue" for value in values]
        costs = [cost + abs(min_cliqNum) for cost in costs]
        cost_plot = plt.bar(values, costs, color=colors)
        plt.text(len(counts)/2-0.5, -0.1*(max_cliqNum+abs(min_cliqNum)), 'Num Nodes: '+str(Graph.getNumNodes())+' Num Edges: '+str(Graph.getNumEdges()), fontsize=15, horizontalalignment='center', verticalalignment="bottom", bbox=dict(facecolor='white', alpha=0.5))
        for val in max_vals:
            index = 0
            for value,_ in sortedCounts:
                if(value == val):   
                    break
                index += 1
            plt.text(index-0.4 - len(str(val))/(2*len(counts)), max_cliqNum + abs(min_cliqNum), str(val), fontsize=8)
        plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
        plt.xlabel("Output States")
        plt.ylabel("Cost offset by lowest Cost")
        plt.title("Output State vs. Cost")
        plt.show()
    return fbest, all_counts
    #return expectation
    #return max_cliqNum, currentMaxCost

    

def mapInputSpace(graph):
    gammahist = []
    betahist = []
    zhist = [[0 for x in range(25)] for i in range(25)]
    print(zhist)
    gamma_space = np.linspace(0, 2*np.pi, 25)
    beta_space = np.linspace(0, np.pi, 25)
    row = 0
    col = 0
    
    for gamma in gamma_space:
        col = 0
        for beta in beta_space:
            zhist[row][col] = applyQAOA(gamma, beta, graph) 
            col += 1
            print("Gamma: %s | Beta : %s " % (gamma, beta))
        row += 1
    # print(zhist)
   
    gammahist = np.asarray(gammahist)
    betahist = np.asarray(betahist)
    gammahist, betahist = np.meshgrid(gamma_space, beta_space)
    print(gammahist)
    print(betahist)
    zhist = np.asarray(zhist)
    print(zhist)
    trace = go.Surface(
        x=gamma_space,
        y=beta_space,
        z=zhist)
    data = [trace]
    py.plot(data,filename="test1", auto_open=True, fileopt="overwrite")

    """fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(gammahist, betahist, zhist, cmap=cm.get_cmap('coolwarm'),
                       linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()"""

def gradient(func, params, epsilon, whichParam):
    first = params
    second = params
    first[whichParam] += epsilon
    second[whichParam] -= epsilon
    return func(*first) - func(*second)/(2*epsilon)

def optimize2(graph, epsilon, eta, threshold):
    count = 0
    gamma = 5.2359
    beta = 0.26179
    dgamma = (applyQAOA(gamma + epsilon, beta, graph) - applyQAOA(gamma - epsilon, beta, graph))/(2*epsilon)
    dbeta = (applyQAOA(gamma, beta + epsilon, graph) - applyQAOA(gamma, beta - epsilon, graph))/(2*epsilon)
    flipper = True #Alternate between maxing gamma and maxing beta
    while((abs(dgamma) + abs(dbeta))/2 > threshold):
        if(flipper):
            if (dgamma > 0): 
                gamma = (gamma + (dgamma * eta)) % (2*np.pi)
            elif (dgamma < 0):
                gamma = (gamma - (dgamma * eta)) % (2*np.pi)
            dgamma = (applyQAOA(gamma + epsilon, beta, graph) - applyQAOA(gamma - epsilon, beta, graph))/(2*epsilon)
        else:
            beta = (beta + (dbeta * eta)) % np.pi
            dbeta = (applyQAOA(gamma, beta + epsilon, graph) - applyQAOA(gamma, beta - epsilon, graph))/(2*epsilon)
            
        count+=1
        print("Function run : ", count)
        print("Gamma : %s | Gamma Gradient: %s" % (gamma, dgamma))
        print("Beta : %s | Beta Gradient: %s" % (beta, dbeta))
        flipper = not flipper
    
    print(count)
    return gamma, beta

### gradient ascent optimizer
# graph is graph to optimize over
# epsilon controls how far out the delta is calculated
# eta is learning rate
# threshold is the average of gamma and beta that we will consider a max

def optimize(graph, epsilon, eta, threshold):
    count = 0
    # gamma = 2.00000242
    # beta = 1.9999998
    # gamma = 2.0017074981249996
    # beta = 2.0007869093750004
    beta = 2
    gamma = 2
    gammahist = [gamma, gamma + epsilon, gamma]
    betahist = [beta, beta,  beta + epsilon]
    zhist = [applyQAOA(gamma, beta, graph)[2], applyQAOA(gamma + epsilon, beta, graph)[2],
            applyQAOA(gamma, beta + epsilon, graph)[2]]
    dgamma = (zhist[-2] - zhist[-3])/(epsilon)
    dbeta = (zhist[-1] - zhist[-3] )/(epsilon)
    gradient = [dgamma, dbeta]
    gamma =  (gamma + (dgamma * eta)) % (2*np.pi)
    beta =  (beta + (dbeta * eta)) % np.pi
    while(np.linalg.norm(gradient) > threshold):
        gammahist += [gamma, gamma+epsilon, gamma]
        betahist += [beta, beta, beta+epsilon]
        zhist += [applyQAOA(gamma, beta, graph)[2], applyQAOA(gamma + epsilon, beta, graph)[2], applyQAOA(gamma, beta + epsilon, graph)[2]]
        dgamma = (zhist[-2] - zhist[-3])/(epsilon)
        dbeta = (zhist[-1] - zhist[-3])/(epsilon) 
        gradient = [dgamma, dbeta]  
        gamma =  (gamma + (dgamma * eta)) % (2*np.pi)
        beta =  (beta + (dbeta * eta)) % np.pi
        count+=1
        print("Function run : ", count)
        print("Gamma : %s | Gamma Gradient: %s" % (gamma, dgamma))
        print("Beta : %s | Beta Gradient: %s" % (beta, dbeta))
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    gammahist = np.asarray(gammahist)
    betahist = np.asarray(betahist)
    zhist = np.asarray(zhist)
    surf = ax.plot_surface(gammahist,betahist,zhist, cmap=cm.get_cmap('coolwarm'),
                       linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    return gamma, beta

"""
def superOptimize(graph, epsilon, eta, threshold, numOfTrials):
    maxVal = -graph.getMaxEdges() * graph.getNumEdges()
    maxG = -1
    maxB = -1
    tempVal = 0
    for i in range(numOfTrials):
        tempG, tempB = optimize(graph, epsilon, eta, threshold, i*2*np.pi/numOfTrials, i*np.pi/numOfTrials)
        tempSum = 0
        for i in range(10):
            tempSum += applyQAOA(tempG, tempB, graph)
        tempVal = tempSum/10
        if(tempVal > maxVal):
            maxVal = tempVal
            maxG = tempG
            maxB = tempB
    return maxG, maxB, maxVal"""
        
def main():
    ### If P > 0
    #gamma = []
    #beta = []
    #   
    #for i in range(P):
    #    gamma.append(np.random.uniform(0,2*np.pi))
    #for i in range(P):
    #    beta.append(np.random.uniform(0,np.pi))

  
    
    ###TESTING GRAPH
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

    # mapInputSpace(myGraph)
    #print(classicalMC(myGraph))
    ### Run the algorithm
    #optimize2(myGraph, 0.01, 0.05, 0.05)
    """allQAOACosts = []
    allClassicalCosts = []"""
    myGraph2 = Graph(9)
    print(myGraph2.getNumEdges())
    print(myGraph2.getNumNodes())
    fbest, allcounts = applyQAOA([4.6015625, 0.18702062], myGraph2)
    """allQAOACosts.append(max_cliqNum)
    allClassicalCosts.append(classicalCost)"""

    #print("Expectation Value:", expect)

    """myGraph2 = Graph(5)
    max_cliqNum, classicalCost = applyQAOA([4.6015625, 0.18702062], myGraph2)
    allQAOACosts.append(max_cliqNum)
    allClassicalCosts.append(classicalCost)
    myGraph3 = Graph(6)
    max_cliqNum, classicalCost = applyQAOA([4.6015625, 0.18702062], myGraph3)
    allQAOACosts.append(max_cliqNum)
    allClassicalCosts.append(classicalCost)
    myGraph4 = Graph(8)
    max_cliqNum, classicalCost = applyQAOA([4.6015625, 0.18702062], myGraph4)
    allQAOACosts.append(max_cliqNum)
    allClassicalCosts.append(classicalCost)"""
   

    """myGraph5 = Graph(10)"""

    ### OPTIMIZE
    #bestGamma, bestBeta = optimize(myGraphTest, 0.05, 0.00001, 0.05)
    """res = minimize(applyQAOA,[(1.047198,3.010693)] , method='L-BFGS-B', bounds={(0, 2*np.pi), (0, np.pi)}, options={'disp': True})
    if res.success:
        fitted_params = res.x
        print(fitted_params)
    else:
        raise ValueError(res.message)"""
    # Optimal Gamma: 3.10693359375 Optimal Beta: 2.50830078125
    # This is very likely a local max though.
    # We might want optimize from various start positions and compare results
    # Also need to discuss optimization parameters cause I kind of chose those arbitrarily

    #bestGamma, bestBeta, bestVal = superOptimize(myGraph, 0.1, 0.1, 0.05, 16)
    #bestGamma = 4.6015625
    #bestBeta = 0.18702062766020688
    #print("BestGamma: ", bestGamma, "bestBeta", bestBeta)

    #fbest, allcounts, _ = applyQAOA(bestGamma, bestBeta, myGraph5)
    #print("Optimal Gamma:", bestGamma, "Optimal Beta:", bestBeta)

   

    #fbest, allcounts, _ = applyQAOA(bestGamma, bestBeta, myGraph2)

    #fbest2, allcounts2, _ = applyQAOA(bestGamma, bestBeta, myGraph3)

    #fbest3, allcounts4, _ = applyQAOA(bestGamma, bestBeta, myGraph4)

    """ax = plt.subplot(111)
    ax.bar([4-0.25, 5-0.25,6-0.25, 8-0.25],allQAOACosts,width=0.5,color='b',align='center')
    ax.bar([4+0.25, 5+0.25, 6+0.25, 8+0.25],allClassicalCosts,width=0.5,color='g',align='center')
    ax.legend(('Max Cost from QAOA','Max Cost from Classical Brute Force'))
    plt.xlabel('Number of Nodes')
    plt.ylabel('Cost')
    plt.title('QAOA Max Cost compared to Classical Max Cost')
    plt.show()"""

    ### Make graphs.
    # I'm thinking we hold one variable constant at its maxed value
    # and vary the other and vice versa.
    # Gamma has a larger range than beta. Do we want more data points for gamma than beta?
    # The last page of the worksheet says exactly which graphs we need in our report
    # so make sure we have at least those
    gamma = 3.10693359375
    beta = 2.50830078125
    betas = np.linspace(0, np.pi, 100)
    gammas = np.linspace(0, 2*np.pi, 100)
    varyingBeta = []
    varyingGamma = []
    
    #y = [applyQAOA(gammaa, beta, myGraph) for gammaa in gammas]
    #with open("varyingGamma.txt", 'w') as f:
    #    json.dump(y, f)
        
    #y = [applyQAOA(gamma, betaa, myGraph) for betaa in betas]
    #with open("varyingBeta.txt", 'w') as f:
    #    json.dump(y, f)
           
    #with open("varyingGamma.txt", 'r') as f:
    #    varyingGamma = json.load(f)
    
    #with open("varyingBeta.txt", 'r') as f:
    #   varyingBeta = json.load(f)

    #betaG = plt.plot(betas, varyingBeta)
    #gammaG = plt.plot(gammas, varyingGamma)
    #plt.legend(('Beta Graph', 'Gamma Graph'))
    #plt.xlabel('Beta and Gamma values')
    #plt.ylabel('Expectation Value')
    #plt.title('Expectation Value vs Gamma and Beta')
    #plt.show()
    """
    plt.scatter([4, 6, 6, 8], all_average_costs)
    plt.scatter([4, 6, 6, 8], all_minimum_costs)
    plt.scatter([4, 6, 6, 8], all_maximum_costs)
    plt.legend(('Average Costs', 'Minimum Costs', 'Maximum Costs'))
    plt.xlabel('Number of nodes')
    plt.ylabel('Cost')
    plt.title('Average, Minimum and Maximum Costs')
    plt.show()
    """
    cost_plot = plt.bar(fbest, allcounts)
    plt.xlabel('QAOA Cost/Theoretical Max Cost')
    plt.ylabel('Number of Counts')
    plt.title('QAOA Cost/Theoretical Max Cost vs. Number of Counts ')
    plt.show()

def myMain():
    qc, c, q_input, q_output = makeCircuit(3, 2)
    #print(measureInput(qc,q_input, c))
    test5(qc, q_input, c)  

#myMain()
main()
