class TreeNode:
    def __init__(self, value, discussed=False):
        self.value = value
        self.discussed = discussed
        self.children = []
        self.space = " "

    def add_child(self, child):
        self.children.append(child)

    def to_string_clear(self, level=1):
        ret = "#" * level + self.space + self.value + "\n"
        for child in self.children:
            ret += child.to_string_clear(level + 1)
        return ret
    
    def __str__(self, level=1):
        discussed_str = f"[{self.discussed}]" if self.discussed is not None else ""
        ret = "#" * level + self.space + self.value + discussed_str + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

    @staticmethod
    def from_str(tree_str):
        lines = tree_str.strip().split('\n')
        root = None
        stack = []
        for line in lines:
            level = line.count('#') - 1
            value, discussed = TreeNode.parse_line(line)
            node = TreeNode(value, discussed)
            if level == 0:
                root = node
            else:
                stack[level - 1].add_child(node)
            if level < len(stack):
                stack = stack[:level]
            stack.append(node)
        return root

    @staticmethod
    def parse_line(line):
        parts = line.strip().split('[')
        value = parts[0].strip('#').strip()
        discussed = False
        if len(parts) > 1:
            discussed_str = parts[1].strip(']').strip()
            discussed = discussed_str.lower() == 'true'
        return value, discussed

def find_nearest_leaf_nodes(root, threshold=3, level=1):
    if not root:
        return []
    queue = [root]
    level_leaves = []
    while queue:
        level_size = len(queue)
        for _ in range(level_size):
            node: TreeNode = queue.pop(0)
            if not node.children:
                level_leaves.append(f"{'#' * level} {node.value}")
                if len(level_leaves) >= threshold:
                    return level_leaves
            else:
                for child in node.children:
                    queue.append(child)
        level += 1
    return level_leaves

def get_topic_from_outlint(outline: str, threshold=1):
    """按照广度优先选取要讨论的议题
    todo: 需要解决多层次节点中，需要返回父节点的问题
    """
    try:
        tree = TreeNode.from_str(outline)
        out = find_nearest_leaf_nodes(tree, threshold=threshold)
        return "\n".join(out)
    except:
        import traceback
        traceback.print_exc()
    return outline

# 示例用法
if __name__ == "__main__":
    # 创建树
    # root = TreeNode("root", discussed=True)
    # child1 = TreeNode("child1", discussed=False)
    # child1_child1 = TreeNode("child1_child1", discussed=True)
    # child2 = TreeNode("child2", discussed=False)
    # child2_child2 = TreeNode("child2_child2", discussed=True)
    
    root = TreeNode("中国近现代史", discussed=False)
    child1 = TreeNode("历史背景", discussed=True)
    child1_child1 = TreeNode("近现代社会问题", discussed=False)
    child2 = TreeNode("国共合作", discussed=True)
    child2_child2 = TreeNode("北伐战争", discussed=True)

    root.add_child(child1)
    root.add_child(child2)
    child1.add_child(child1_child1)
    child2.add_child(child2_child2)

    # 输出树
    # print(root)
    # print(root.to_string_clear())
    # print(find_nearest_leaf_nodes(root))

    # 从字符串重建树
    tree_str = """
    # 中国近现代史中各党派作用
    ## 国民党作用
    ### 辛亥革命领导
    ### 建立民国政府
    ### 现代化进程推动
    ### 执政能力质疑
    #### 民主化挑战
    #### 经济改革问题
    ## 共产党作用
    ## 民主党派作用
    ## 其他小党派影响
    """
    reconstructed_tree = TreeNode.from_str(tree_str)
    print(reconstructed_tree)
    print(find_nearest_leaf_nodes(reconstructed_tree, threshold=10))
