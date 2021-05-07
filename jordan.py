import sympy

def generalized_eigenvector_decomp(adj, deg="to", omit_first=False, return_J=False, exclude_general=False):
    """Computes the generalize eigenvector decomposition of a graph given
its adjacency matrix.  These values can be complex.

Parameters
----------
adj: torch.tensor
    The adjacency matrix of the graph.  Must not be sparse
deg: str, torch.tensor. optional
    The degree matrix of the graph, or the string "to" or "from"
    indicating that the to- or from-adjacency matrix (resp.) should be
    used to compute the laplacian.  Defalts to "to"
omit_first: bool. optional
    If True, excludes the first generalized eigenvector.  Defaults to
    False.
return_J: bool. optional
    If True, return the jordan canonical form of the laplace matrix.
    Defaults to False
exclude_general: bool.
    If True, remove non-true eigenvectors (generalized).  Defaults to
    False

Returns
-------
P: sympy.Matrix
    The eigenvector decomposition of each node where each column
    represents the corresponding node (column 0 <-> node 0)
J: sympy.Matrix, None
    If return_J is true, J is the jordan cannonical form of the
    laplace matrix.  If eigenvectors were excluded from P (because
    exclude_general and/or omit_first were true), then rows and
    columns corresponding to the same eigenvectors are excluded from J

    """
    # obtain adjacecy matrix
    if deg == "to":
        deg = sympy.diag(*(v.item() for v in A.sum(0)))
    elif deg == "from":
        deg = sympy.diag(*(v.item() for v in A.sum(1)))
    else:
        deg = sympy.Matrix(deg)

    # obtain laplacian
    A = sympy.Matrix(A)
    L = deg-A

    # calculate generalized eigenvectors and jordan canonical form
    P, J = L.jordan_form()

    # sort by eigenvalue smallest first
    for i in range(J.shape[0]-1):
        for j in range(J.shape[0]-i-1):
            if J[j,j] > J[j+1,j+1]:
                J.row_swap(j, j+1)
                J.col_swap(j, j+1)
                P.row_swap(j, j+1)

    # first eigenvalue is usuall omitted
    if omit_first:
        P.row_del(0)
        J.row_del(0)
        J.col_del(0)

    # exclude generalized eigenvectors
    if exclude_general:
        exclude = []
        for i in range(J.shape[0]-1):
            if J[i, i+1] == 1:
                exclude.insert(0, i)
        for i in exclude:
            P.row_del(i)
            J.row_del(i)
            J.col_del(i)

    # return decomposition
    if return_J:
        return P, J
    return P
