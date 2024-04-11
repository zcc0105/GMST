import os
import copy
import numpy as np
import networkx as nx
import random

def sequenceRead(filename, length):
    sequenceList = []
    i = 0
    with open(filename, 'r') as f:
        for line in f.readlines():
            if i >= length:
                continue
            data = line.split()
            sequenceList.append(data)
            i = i + 1
        f.close()
    return sequenceList


def indexSearch(data: list, i_):
    index = []
    for i in range(len(data)):
        if data[i] == i_:
            index.append(i)
    return index


class CompareAlg:
    def __init__(self, graph, opts):
        self.graph = graph
        self.opts = opts
        self.address = os.getcwd() + '/MC_nodes/' + opts.dataset_name + str(opts.seeds_size) + '.txt'
        self.influ_seeds = {}

    def loadGraph(self, seeds, compSeeds, candSeeds):
        self.graph.graph['free'] = []
        self.graph.graph['1'] = list(seeds)
        self.graph.graph['2'] = list(compSeeds)
        self.graph.graph['3'] = list(candSeeds)
        for node in self.graph.nodes():
            if node not in self.graph.graph['1'] and node not in self.graph.graph['2'] and node not in self.graph.graph['3']:
                self.graph.graph['free'].append(node)
        return

    def activate(self, node, player):
        if node in self.graph.graph['free']:
            self.graph.graph['free'].remove(node)
        self.graph.graph[str(player)].append(node)
        return

    def weight(self, u, v):
        if self.graph.has_edge(u, v):
            return self.graph[u][v]['weight']
        else:
            return 0

    def getActivate(self, u, active1, active2):
        if active1 == 0 and active2 == 0:
            return 0
        elif active1 > active2:
            self.activate(u, 1)
            return 1
        elif active1 < active2:
            self.activate(u, 2)
            return 2
        else:
            k = np.random.randint(1, 3)
            self.activate(u, k)
            return k

    def compDiffuseIndependentCascade(self, seeds, compSeeds):
        iteration = 0
        iterNumber = 2
        nodeThreshold = {}

        newActive1 = True
        newActive2 = True

        currentActiveSeeds = copy.deepcopy(seeds)
        currentActiveCompSeeds = copy.deepcopy(compSeeds)

        newActiveSeeds = set()
        newActiveCompSeeds = set()

        activeNodes = copy.deepcopy(seeds)
        activeNodes.extend(copy.deepcopy(compSeeds))
        while newActive1 and newActive2 and iteration < iterNumber:
            # while newActive1:
            iteration = iteration + 1
            judgeUnionSeeds = set()
            judgeUnionCompSeeds = set()

            for node in currentActiveSeeds:
                for neighbor in self.graph.neighbors(node):
                    judgeUnionSeeds.add(neighbor)
            for node in currentActiveCompSeeds:
                for neighbor in self.graph.neighbors(node):
                    judgeUnionCompSeeds.add(neighbor)
            activeNodes_ = set(activeNodes)
            unActiveUnionNode = (judgeUnionSeeds & judgeUnionCompSeeds) - (
                    judgeUnionSeeds & judgeUnionCompSeeds & activeNodes_
            )
            unActiveNodes = ((judgeUnionSeeds | judgeUnionCompSeeds) - (judgeUnionSeeds & judgeUnionCompSeeds)) - (
                    (judgeUnionSeeds | judgeUnionCompSeeds) & activeNodes_
            )
            if not unActiveNodes:
                newActive1 = False
                continue
            if np.random.randint(1, 3) == 1:
                isParty1 = True
            else:
                isParty1 = False
            t = 0
            while t < 2:
                t = t + 1
                if isParty1:
                    isParty1 = False
                    delnodes = []
                    for node in unActiveNodes:
                        if np.random.uniform(0, 1) >= 0.5:
                            self.activate(node, 1)
                            newActiveSeeds.add(node)
                            activeNodes.append(node)
                            delnodes.append(node)
                    unActiveNodes = list(set(unActiveNodes) - set(delnodes))
                else:
                    isParty1 = True
                    delnodes = []
                    for node in unActiveNodes:
                        if np.random.uniform(0, 1) >= 0.5:
                            self.activate(node, 2)
                            newActiveCompSeeds.add(node)
                            activeNodes.append(node)
                            delnodes.append(node)
                    unActiveNodes = list(set(unActiveNodes) - set(delnodes))
            for node in unActiveUnionNode:
                if np.random.uniform(0, 1) >= 0.5:
                    k = np.random.randint(1, 3)
                    if k == 2:
                        self.activate(node, 2)
                        newActiveCompSeeds.add(node)
                        activeNodes.append(node)
                    elif k == 1:
                        self.activate(node, 1)
                        newActiveSeeds.add(node)
                        activeNodes.append(node)
            activeNodes = list(set(activeNodes).difference(self.graph.graph['3']))
            newActiveSeeds = set(newActiveSeeds).difference(self.graph.graph['3'])
            newActiveCompSeeds = set(newActiveCompSeeds).difference(self.graph.graph['3'])

            if newActiveSeeds:
                currentActiveSeeds = list(newActiveSeeds)
                newActiveSeeds = set()
            else:
                newActive1 = False
            if newActiveCompSeeds:
                currentActiveCompSeeds = list(newActiveCompSeeds)
                newActiveCompSeeds = set()
            else:
                newActive2 = False
        return len(self.graph.graph['1']) + len(self.graph.graph['2'])

    def spreadCompute(self, args):
        seeds = args[0]
        compSeeds = args[1]
        candSeeds = args[2]
        self.loadGraph(seeds, compSeeds, candSeeds)
        spread = self.compDiffuseIndependentCascade(seeds, compSeeds)
        return spread

    def spreadGet(self, seeds, compSeeds, candSeeds, isPool=True):
        iterations = 50
        if isPool:
            from multiprocessing import Pool
            cpu_worker_num = 8
            process_args = []
            for _ in range(iterations):
                process_args.append((seeds, compSeeds, candSeeds))
            with Pool(cpu_worker_num) as p:
                spreads = p.map(self.spreadCompute, process_args)
            return np.average(spreads)

        else:
            args = (seeds, compSeeds, candSeeds)
            influence = []
            for _ in range(iterations):
                influence.append(self.spreadCompute(args))
            return np.average(influence)

    def list_gather(self, list_seeds):
        condition = lambda seed: seed in list_seeds
        res = [[] for _ in range(self.opts.iterations)]
        for key, value in self.influ_seeds.items():
            if condition(key):
                for i in range(self.opts.iterations):
                    res[i].extend(self.influ_seeds[key][i])
        return round(np.average([len(set(sublist)) for sublist in res]), 2)

    def spreadReturn(self, seeds, cSeeds):
        merged_values_seeds = self.list_gather(seeds)
        merged_values_cSeeds = self.list_gather(cSeeds)
        return round(merged_values_seeds+merged_values_cSeeds, 2)

    def multiPool(self, seeds, compSeeds, candSeeds):
        iterations = self.opts.iterations
        if self.opts.isPool:
            from multiprocessing import Pool
            cpu_worker_num = 8
            process_args = []
            for _ in range(iterations):
                process_args.append((seeds, compSeeds, candSeeds))
            with Pool(cpu_worker_num) as p:
                influ_total = p.map(self.CIC, process_args)
            return influ_total
        else:
            args = (seeds, compSeeds, candSeeds)
            influ_total = []
            for _ in range(iterations):
                influ_total.append(self.CIC(args))

            return influ_total

    def CIC(self, args):
        seeds = args[0]
        compSeeds = args[1]
        candSeeds = args[2]
        active_nodes = set(seeds + compSeeds)
        new_active_nodes = set(copy.deepcopy(active_nodes))

        while True:
            current_new_active_nodes = copy.deepcopy(new_active_nodes)
            for node in current_new_active_nodes:
                if isinstance(self.graph, nx.Graph):
                    neighbors = self.graph.neighbors(node)
                else:
                    neighbors = self.graph.successors(node)
                for neighbor in neighbors:
                    if neighbor not in active_nodes and neighbor not in candSeeds:
                        if np.random.uniform(0, 1) <= self.graph.nodes[neighbor]['node_object'].activate_pro:  # 先激活
                            new_active_nodes.add(neighbor)
                            active_nodes.add(neighbor)
            new_active_nodes = new_active_nodes.difference(current_new_active_nodes)
            if not new_active_nodes:
                break

        return len(active_nodes)

    def influGet(self, order):     # sequence (list) ; dataset_id (int)
        inf = []
        sequenceList = sequenceRead("seqResListDe50.txt", self.opts.seq_length)
        i = len(order)
        for seq in sequenceList:
            candSeeds = [order[j] for j in indexSearch(seq[:i], '0')]
            seeds = [order[j] for j in indexSearch(seq[:i], '1')]
            compSeeds = [order[j] for j in indexSearch(seq[:i], '2')]
            inf.append(self.spreadGet(seeds, compSeeds, candSeeds))
        return round(np.average(inf), 3)

    def GreedyMST(self, sequence):
        seqlen = len(sequence)
        root_index = sequence[np.random.randint(0, seqlen)]
        weight = {}
        for node in sequence:
            for node_ in sequence:
                if node == root_index:
                    weight[(node, node_)] = self.influGet([node])
                else:
                    weight[(node, node_)] = -np.inf
        T, T_path, weight = self.prim(weight, sequence, root_index=root_index)
        L = self.preorder_tree_walk(T, sequence, root_index=root_index)
        L.append(root_index)
        H_path = self.create_H(weight, L, sequence)
        print(H_path)
        return H_path, self.influGet(H_path)

    def find_max_edge(self, weight, visited_ids, no_visited_ids):
        max_weight, max_from, max_to = -np.inf, -np.inf, -np.inf
        for from_index in visited_ids:
            for to_index in no_visited_ids:
                if from_index != to_index and weight[
                    (from_index, to_index)] > max_weight and to_index in no_visited_ids:
                    max_to = to_index
                    max_from = from_index
                    max_weight = weight[(max_from, max_to)]
        return (max_from, max_to), max_weight

    def contain_no_visited_ids(self, sequence, visited_ids):
        no_visited_ids = []
        [no_visited_ids.append(node) for node in sequence if node not in visited_ids]
        return no_visited_ids

    def prim(self, weight, sequence: list, root_index):
        solution = [root_index]
        T_path = []
        seqlen = len(sequence)
        no_visit_nodes = sequence[:]
        no_visit_nodes.remove(root_index)
        while len(solution) != seqlen:
            (max_from, max_to), max_weight = self.find_max_edge(weight, solution, no_visit_nodes)
            solution.append(max_to)
            T_path.append((max_from, max_to))
            no_visit_nodes.remove(max_to)
            # 计算weight
            value = self.influGet(solution)
            for node in no_visit_nodes:
                list_ = solution[:]
                list_.append(node)
                weight[(max_to, node)] = self.influGet(list_) - value
        T = np.full((len(sequence), len(sequence)), -np.inf)
        for (from_, to_) in T_path:
            index1 = solution.index(to_)
            index2 = solution.index(from_)
            T[index2][index1] = weight[(from_, to_)]
            list1 = solution[index1:]
            list2 = solution[index2:]
            T[index1][index2] = np.inf
            weight[(to_, from_)] = T[index1][index2]
        return T, T_path, weight

    def preorder_tree_walk(self, T, sequence, root_index):
        is_visited = [False] * T.shape[0]
        stack = [root_index]
        T_walk = []
        while len(stack) != 0:
            node = stack.pop()
            T_walk.append(node)
            is_visited[sequence.index(node)] = True
            nodes_index = np.where(T[sequence.index(node)] != -np.inf)[0]
            if len(nodes_index) > 0:
                [stack.append(sequence[index_]) for index_ in reversed(nodes_index) if is_visited[index_] is False]
        return T_walk

    def create_H(self, weight, L, sequence):
        H_path = []
        for i, from_node in enumerate(L[0:-1]):
            H_path.append(from_node)
        # return H, H_path
        return H_path

    def maxSeq(self, list_, seqlen):
        index = 0
        res = {}
        max_seq = []
        max_value = 0.0
        while index < seqlen:
            seq = []
            for node in list_[index:]:
                seq.append(node)
            for node in list_[0:index]:
                seq.append(node)
            value = self.influGet(seq)
            if value > max_value:
                max_value = value
                max_seq = seq
            index = index + 1
        return max_seq, max_value

    def caseStudy(self, sequences):
        res = []
        for sequence in sequences:
            curr_seq = []
            for node in sequence:
                next_seq = curr_seq[:]
                next_seq.append(node)
                res_value = self.influGet(next_seq) - self.influGet(curr_seq)
                print(res_value)
                res.append(res_value)
                curr_seq.append(node)
        return res






