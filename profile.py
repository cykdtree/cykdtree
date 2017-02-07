from cykdtree.tests import test_kdtree
import cProfile, pstats

cProfile.run('test_kdtree.time_tree_construction(int(1e6), 10)','restats')
#cProfile.run('test_kdtree.time_neighbor_search(int(1e6), 10)','restats')
p = pstats.Stats('restats')

p.sort_stats('tottime').print_stats(10)
