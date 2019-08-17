from mytree import Node

#recursively flip tree
def invertTree(node):
	if node.left != None:
		invertTree(node.left)
	if node.right != None:
		invertTree(node.right)

	if node.right!= None:
		temp = node.right
		node.right = node.left
		node.left = temp
	return 


dude = Node(8)
dude.insert(6)
dude.insert(9)
dude.insert(2)
dude.insert(7)
dude.insert(2)
dude.insert(90)
dude.prettyPrintTree()
invertTree(dude)
dude.prettyPrintTree()



