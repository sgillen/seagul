class Node():
    def __init__(self,data):
        self.data = data
        self.next = None

class Llist():
    def __init__(self):
        pass
        self.root = None
    def append(self, node):
        if self.root is None:
            self.root = node

        cur = self.root
        while cur.next is not None:
            cur = cur.next

        cur.next = node

    def __repr__(self):
        if self.root is None:
            return 'Empty Llist'

        cur = self.root
        while cur.next is not None:
            print(cur.data)
            cur = cur.next



