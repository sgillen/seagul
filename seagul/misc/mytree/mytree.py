#my binary tree

#the Node class IS the tree, each Node is the root of it's own tree
class Node:
	def __init__(self, value):
		self.left = None
		self.right = None
		self.value = value

	#this is a binary search tree, we assume if we insert a repeated value we want it on the right
	def insert(self, value):
		if self.value == None:
			self.value = value

		if  value < self.value:
			if self.left == None:
				self.left = Node(value)
			else:	
				self.left.insert(value)

		if value >= self.value :
			if self.right == None:
				self.right = Node(value)
			else:
				self.right.insert(value)

	#prints all the elements of the tree rather dumbly
	def printTree(self):
		if self.left != None:
			self.left.printTree()
		if self.right != None:
			self.right.printTree()

		print self.value
		return			

	#finds the max depth, used by the pretty print function
	def findDepth(self):
		left_depth = 0
		right_depth = 0

		if self.left != None:
			left_depth = self.left.findDepth()
		if self.right != None:
			right_depth = self.right.findDepth()

		greater_depth = left_depth if left_depth > right_depth else right_depth

		return greater_depth + 1


	#prints the tree in a much more intuitive and visually pleasing manner
	#TODO: large numbers often mess with this
	def prettyPrintTree(self):
		depth = self.findDepth()
		#in python an "array" is just a list of lists
		tree_array = [["" for i in range(3*depth)] for j in range(2*depth)]

		self._printNode(tree_array,0,depth*3/2)


		for i in range(2*depth):
			print
			for j in range(3*depth):
				print tree_array[i][j],

	#recursively prints the tree, but needs to be called by prettyPrintTree 
	def _printNode(self,tree_array,i,j):

		tree_array[i][j] = self.value

		if self.left != None:
			tree_array[i+1][j-1] = "/"
			self.left._printNode(tree_array, i+2, j-2)
		if self.right != None:
			tree_array[i+1][j+1] = "\\"
			self.right._printNode(tree_array, i+2, j+2)

		return 




		


		










