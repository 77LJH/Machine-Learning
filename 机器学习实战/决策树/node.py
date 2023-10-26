class Node:
    def __init__(self, divide_children=None, divide_attribute=None, classification=None, is_leaf=False):
        if divide_children is None:
            divide_children = {}
        self.divide_children = divide_children  # 字典，key为划分属性的值，value划分后的子节点
        self.divide_attribute = divide_attribute
        self.classification = classification
        self.is_leaf = is_leaf

    def __str__(self):
        if self.is_leaf:
            return f"Leaf Node: Classification = {self.classification}"
        else:
            return f"Internal Node: Divide Attribute = {self.divide_attribute}, Divide Children = {self.divide_children}"



