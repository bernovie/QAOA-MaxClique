import numpy as np

class Edge:
	def __init__(self, node1, node2):
		self.nodes = [node1, node2]

	def connects(self, node):
		return node in self.nodes

	def getNodes(self):
                return self.nodes

	def __eq__(self, otherEdge):
		for node in self.nodes:
			if(not node in otherEdge.nodes):
				return False
		return True
	
	def __ne__(self, otherEdge):
		return not self == otherEdge

class Node:
	def __init__(self, name):
		self.name = name
		self.edges = []
	
	def addEdge(self, edge):
		if (edge.connects(self)):
			self.edges.append(edge)
	
	def __eq__(self, node):
		if(self.name == node.name):
			return True
		return False
	def __ne__(self, node):
		return not self == node

class Graph:
	def __init__(self, numNodes, numEdges=None):
		self.maxEdges = numNodes * (numNodes - 1) /2
		if numEdges == None:
			self.numEdges = np.random.randint(1, self.maxEdges+1)
		else:
			self.numEdges = numEdges
		self.numNodes = numNodes
		self.edges = []
		self.nodes = []
		for i in range(self.numNodes):
			self.nodes.append(Node(i))
		if(numNodes > 1):
                        for i in range(self.numEdges):
                                node1 = self.nodes[np.random.randint(0, self.numNodes)]
                                node2 = self.nodes[np.random.randint(0, self.numNodes)]
                                while(node1 == node2):
                                        node2 = self.nodes[np.random.randint(0, self.numNodes)]
                                myEdge = Edge(node1, node2)
                                while(myEdge in self.edges):
                                        node1 = self.nodes[np.random.randint(0, self.numNodes)]
                                        node2 = self.nodes[np.random.randint(0, self.numNodes)]
                                        while(node1 == node2):
                                                node2 = self.nodes[np.random.randint(0, self.numNodes)]
                                        myEdge = Edge(node1, node2)
                                self.edges.append(myEdge)

	def getEdgesComp(self):
		allEdges = []
		edgesComplement = []
		for i in range(self.numNodes):
			for j in range(i+1, self.numNodes):
				allEdges.append(Edge(self.nodes[i], self.nodes[j]))

		for edge in allEdges:
			if not edge in self.edges:
				edgesComplement.append(edge)
		return edgesComplement

	def getMaxEdges(self):
		return int(self.numNodes*(self.numNodes - 1)/2)

	def getEdges(self):
		return self.edges

	def getNodes(self):
		return self.nodes

	def getNumEdges(self):
		return self.numEdges

	def getNumNodes(self):
		return self.numNodes

	def addEdge(self, edge):
		if (not edge in self.edges):
			self.edges.append(edge)
			self.numEdges += 1

	def addNode(self, node):
		if (not node in self.nodes):
			self.nodes.append(node)
			self.numNodes += 1

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

	def __ne__(self, otherGraph):
		return not self == otherGraph
