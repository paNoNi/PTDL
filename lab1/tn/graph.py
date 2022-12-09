class Graph:
    def __init__(self):
        self.first_nodes = list()
        self.last_nodes = list()

    def forward(self):
        pass


class Link:
    def __init__(self):
        self.local_grad = None
        self.prev_node = None
        self.next_node = None


class Node:
    def __init__(self, value: int):
        self.value = value
        self.grad = None
        self.operation = None
        self.left_node, self.right_node, result_node = None, None, None
        self.graph = None

    def __add__(self, node):
        new_node = Node(self.value + node.value)
        new_node.operation = '+'
        left_link, right_link = Link(), Link()
        left_link.prev_node, right_link.prev_node = self, node
        left_link.next_node, right_link.next_node = new_node, new_node
        return new_node

    def __mul__(self, node):
        new_node = Node(self.value * node.value)
        new_node.operation = '*'
        left_link, right_link = Link(), Link()
        left_link.prev_node, right_link.prev_node = self, node
        left_link.next_node, right_link.next_node = new_node, new_node
        return new_node
