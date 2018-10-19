import numpy as np

class Edge:
    def __init__(self, node1, node2):
        self.nodes = [node1, node2]

    def connects(node):
        return node in self.nodes
    
    def __eq__(self, otherEdge):
        for node in self.nodes:
	    if(not node in otherEdge.nodes):
	        return False
        return True
    
    def __ne__(self, otherEdge):
        return not __eq__(otherEdge)

class Node:
    def __init__(self, name):
        self.name = name
        self.edges = []
	
    def addEdge(self, edge):
        if (edge.connects(self)):
        	self.edges.append(edge)
    
    def __eq__(self, node):
        if(self.id == node.id):
	    return True
	return False
    def __ne__(self, node):
        return not __eq__(node)

class Graph:
    def __init__(self, numNodes):
    	self.maxEdges = numNodes * (numNodes - 1) /2
        self.numEdges = np.random.randint(0, self.maxEdges)
	self.numNodes = numNodes
	self.edges = []
	self.nodes = []
	for i in range(self.numNodes):
	    nodes.append(Node(i))
	
	for i in range(self.numEdges):
	    while(True):
	        while(True):
	            node1 = self.nodes[np.random.randint(0, self.numNodes)]
	            node2 = self.nodes[np.random.randint(0, self.numNodes)]
	            if (node2 != node1):
		        break
	        edge1 = Edge(node1, node2)
		if (not edge1 in self.edges):
		    break


    def __init__(self, numNodes, numEdges):
    	self.maxEdges = numNodes * (numNodes - 1) /2
        self.numEdges = numEdges
	self.numNodes = numNodes
	self.edges = []
	self.nodes = []
	for i in range(self.numNodes):
	    nodes.append(Node(i))
	
	for i in range(self.numEdges):
	    while(True):
	        while(True):
	            node1 = self.nodes[np.random.randint(0, self.numNodes)]
	            node2 = self.nodes[np.random.randint(0, self.numNodes)]
	            if (node2 != node1):
		        break
	        edge1 = Edge(node1, node2)
		if (not edge1 in self.edges):
		    break
        
    def maxEdges(self):
        return self.maxEdges

    def edges(self):
        return self.edges

    def nodes(self):
        return self.nodes

    def numEdges(self):
        return self.numEdges

    def numNodes(self):
        return self.numNodes

    def addEdge(self, edge):
        if (not edge in self.edges):
	    self.edges.append(edge)
	    self.numEdges++

    def addNode(self, node):
        if (not node in self.nodes):
	    self.nodes.append(node)
	    self.numNodes++

    def __eq__(self, otherGraph):
        if(self.numNodes != otherGraph.numNodes):
	    return False
	if(self.numEdges != otherGraph.numEdges):
	    return False
	for node in self.nodes:
	    if(not node in otherGraph.nodes):
	        return False
        for edge in self.edges:
	    if(not edge in otherGraph.edges):
	        return False
        return True
