mpirun -n 4 python -c 'from cykdtree.tests.test_parallel_kdtree import *; test_PyParallelKDTree()'
mpirun -n 4 python -c 'from cykdtree.tests.test_parallel_kdtree import *; test_PyParallelKDTree_errors()'
mpirun -n 4 python -c 'from cykdtree.tests.test_parallel_kdtree import *; test_neighbors()'
mpirun -n 4 python -c 'from cykdtree.tests.test_parallel_kdtree import *; time_tree_construction(1e6, 10, 2)'
