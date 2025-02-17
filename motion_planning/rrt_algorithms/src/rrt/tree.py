from rtree import index

# https://rtree.readthedocs.io/en/latest/tutorial.html
class Tree(object):
    def __init__(self, X):
        """
        Tree representation
        :param X: Search Space
        """
        p = index.Property()
        p.dimension = X.dimensions
        self.V = index.Index(interleaved=True, properties=p)  # vertices in an rtree
        self.V_count = 0
        self.V_all = []
        self.E = {}  # edges in form E[child] = parent
        self.E_samples = {}  # samples for each edge

        # real distance and gradient, used for variable
        self.E_dis_grad = {}
