import os
import copy
from seedAndGraphInfo import SeedsInfo
import pprint
from options import get_options
import time
from comAlg import CompareAlg


def run(opts):
    pprint.pprint(vars(opts))
    seedObject = SeedsInfo()
    datasetFileName = os.getcwd() + './dataset/' + opts.dataset_name + '.txt'
    print(datasetFileName)
    graph = seedObject.parse_graph_txt_file(datasetFileName)
    seed_index = int(opts.seeds_size / 10 - 1)
    print(seed_index)
    seeds = copy.deepcopy(seedObject.seeds[opts.dataset_name][seed_index])
    seqDict = {}
    for i in range(8):
        seqDict[i] = seeds[:]

    CAlg_object = CompareAlg(graph, opts)

    # Random algorithm
    if opts.is_Random:
        print('Random is running')
        startTime = time.time()
        print(CAlg_object.Random(seqDict[0]))
        endTime = time.time()
        print('Running time: ' + str(endTime - startTime))

    # High Degree algorithm
    if opts.is_HG:
        print('HG is running')
        startTime = time.time()
        print(CAlg_object.HighDegree(seqDict[1]))
        endTime = time.time()
        print('Running time: ' + str(endTime - startTime))

    # Greedy algorithm
    if opts.is_Greedy:
        print('Greedy is running')
        startTime = time.time()
        print(CAlg_object.greedy(seqDict[2]))
        endTime = time.time()
        print('Running time: ' + str(endTime-startTime))

    # RGreedy algorithm
    if opts.is_RG:
        print('RG is running')
        startTime = time.time()
        print(CAlg_object.RGreedy(seqDict[3]))
        endTime = time.time()
        print('Running time: ' + str(endTime-startTime))

    # GreedyMST
    if opts.is_GMST:
        print('GMST is running')
        startTime = time.time()
        print(CAlg_object.GreedyMST(seqDict[4]))
        endTime = time.time()
        print('Running time: ' + str(endTime-startTime))

    # GreedyMST
    if opts.is_PRank:
        print('PageRank is running')
        startTime = time.time()
        print(CAlg_object.RageRank(seqDict[5]))
        endTime = time.time()
        print('Running time: ' + str(endTime - startTime))

    if opts.is_Close:
        print('Closeness is running')
        startTime = time.time()
        print(CAlg_object.Closeness(seqDict[6]))
        endTime = time.time()
        print('Running time: ' + str(endTime - startTime))

    if opts.is_Between:
        print('Betweenness is running')
        startTime = time.time()
        print(CAlg_object.Betweenness(seqDict[7]))
        endTime = time.time()
        print('Running time: ' + str(endTime - startTime))

    sequences = [
        [3297, 3298, 3299, 3300, 823, 92, 837, 89, 822, 563],
        [92, 823, 563, 3297, 3298, 3299, 3300, 837, 89, 822]
    ]
    print(CAlg_object.caseStudy(sequences))

if __name__ == '__main__':
    run(get_options())