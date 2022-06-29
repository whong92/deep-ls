import numpy as np
import cv2
import copy

from gym import Env

from deepls.graph_utils import tour_nodes_to_tour_len, tour_nodes_to_W, tour_nodes_to_node_rep
from deepls import plot_utils
from deepls.google_tsp_reader import GoogleTSPReader
import matplotlib.pyplot as plt

font = cv2.FONT_HERSHEY_COMPLEX_SMALL


def compute_dist(u, v, sqrt: bool = False):
    dist = (u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2
    if sqrt:
        return np.sqrt(dist)
    return dist


# quick mafs for neighbor computation
def compute_neighbor_delta(tour: np.array, coords: np.array, i, k):
    assert k - i >= 2
    N = len(tour)
    u, v = tour[i], tour[i+1]
    x, y = tour[k], tour[(k + 1) % N]
    return -(compute_dist(coords[u], coords[v], True) + compute_dist(coords[x], coords[y], True)) + \
           (compute_dist(coords[x], coords[u], True) + compute_dist(coords[v], coords[y], True))


def perform_swap(tour_nodes: np.array, i, k, adj):
    # k larger than i, and k > i + 1, ie the two edges do not share a node
    N = len(tour_nodes)
    assert k - i >= 2 and N - k + i >= 2

    # len v -> x
    nvx = k - (i + 1)
    # len y -> u
    if (k + 1) % N == (k + 1):
        nyu = i + N - (k + 1)
    else:
        nyu = i - (k + 1) % N

    # smaller wraparound, rotate the tour, this creates a new array
    if nvx > nyu:
        tour_nodes = np.concatenate((tour_nodes[k:], tour_nodes[:k]))
        i_old = i
        i = 0
        k = i_old + len(tour_nodes[k:])

    if k == N - 1:
        tour_nodes = np.concatenate((tour_nodes[k:], tour_nodes[:k]))
        i_old = i
        i = 0
        k = i_old + len(tour_nodes[k:])

    # the segment u -> y
    segment = tour_nodes[i:k + 2]

    # TODO: this is for a directed adj - fix for undirected
    u, v = tour_nodes[i], tour_nodes[(i + 1) % N]
    x, y = tour_nodes[k], tour_nodes[(k + 1) % N]
    # remove old edges
    adj[u, v] = 0
    adj[x, y] = 0
    adj[v, u] = 0
    adj[y, x] = 0

    # reverse x -> v
    nodes_to_rev = segment[1:-1]
    tour_nodes[i + 1:k + 1] = nodes_to_rev[::-1]

    # TODO: this is for a directed adj - fix for undirected
    # add new edges
    adj[u, x] = 1
    adj[x, u] = 1
    adj[v, y] = 1
    adj[y, v] = 1

    return tour_nodes


class TSP2OptState:
    def __init__(self, nodes_coord: np.ndarray, edge_weights: np.ndarray, init_tour: np.ndarray, opt_tour_len: int):
        self.num_nodes = nodes_coord.shape[0]
        self.nodes_coord = nodes_coord
        self.edge_weights = edge_weights
        assert init_tour.shape[0] == self.num_nodes
        self.tour_nodes = init_tour.copy()
        self.nodes_pos = tour_nodes_to_node_rep(self.tour_nodes)
        self.tour_adj = tour_nodes_to_W(self.tour_nodes)
        self.tour_len = tour_nodes_to_tour_len(self.tour_nodes, edge_weights)
        self.opt_tour_len = opt_tour_len

    def apply_move(self, e1: np.ndarray, e2: np.ndarray):
        i, j = e1
        k, l = e2

        assert self.tour_adj[i, j] == 1 and self.tour_adj[k, l] == 1
        assert self.tour_adj[j, i] == 1 and self.tour_adj[l, k] == 1
        pos_i = self.nodes_pos[i]
        pos_j = self.nodes_pos[j]
        # always make sure the edge is facing forwards in the tour, including the wraparound
        # edge
        if pos_i == 0 and pos_j == self.num_nodes - 1:
            i, j = j, i
            pos_i, pos_j = pos_j, pos_i
        elif pos_i > pos_j:
            if not (pos_i == self.num_nodes - 1 and pos_j == 0):
                i, j = j, i
                pos_i, pos_j = pos_j, pos_i

        pos_k = self.nodes_pos[k]
        pos_l = self.nodes_pos[l]
        # always make sure the edge is facing forwards in the tour, including the wraparound
        # edge
        if pos_k == 0 and pos_l == self.num_nodes - 1:
            k, l = l, k
            pos_k, pos_l = pos_l, pos_k
        elif pos_k > pos_l:
            if not (pos_k == self.num_nodes - 1 and pos_l == 0):
                k, l = l, k
                pos_k, pos_l = pos_l, pos_k

        if pos_k < pos_i:
            i, j, k, l = k, l, i, j
            pos_i, pos_j, pos_k, pos_l = pos_k, pos_l, pos_i, pos_j

        N = self.num_nodes
        # following check implies both edges share a node
        if pos_k - pos_i < 2 or N - pos_k + pos_i < 2:
            # we're done
            return

        # perform swap
        self.tour_len += compute_neighbor_delta(self.tour_nodes, self.nodes_coord, pos_i, pos_k)
        self.tour_nodes = perform_swap(self.tour_nodes, pos_i, pos_k, self.tour_adj)
        self.nodes_pos = tour_nodes_to_node_rep(self.tour_nodes)


class TSP2OptEnv(Env):
    def __init__(
        self,
        num_nodes=10,
        data_f='../graph-convnet-tsp/data/tsp10_val_concorde.txt',
        max_num_steps=50,
        shuffle_data=True,
        ret_best_state=True,
        ret_log_tour_len=False,
        seed=42
    ):
        super(TSP2OptEnv, self).__init__()
        # config vars
        self.max_num_steps = max_num_steps
        self.num_nodes = num_nodes
        self.data_f = data_f
        self.shuffle_data = shuffle_data
        self.seed = seed
        self.ret_best_state = ret_best_state
        self.ret_log_tour_len = ret_log_tour_len
        self.init()

    def init(self):
        self.reader = GoogleTSPReader(self.num_nodes, -1, 1, self.data_f, shuffle_data=self.shuffle_data)
        self.reader_iter = self.reader.__iter__()
        self.cur_step = -1

        self.cur_instance = None
        self.tour_nodes = None
        self.rng = np.random.default_rng(self.seed)

    def reset(self, fetch_next=True):
        if fetch_next or self.cur_instance is None:
            # get a new batch from TSPReader and initialize the state
            try:
                self.cur_instance = next(self.reader_iter)
            except StopIteration as e:
                self.reader_iter = self.reader.__iter__()
                self.cur_instance = next(self.reader_iter)

        b = self.cur_instance
        tour_nodes = np.arange(len(b['nodes_coord'][0]), dtype=int)  # greedy_search(b['nodes_coord'][0])
        self.state = TSP2OptState(b['nodes_coord'][0], b['edges_values'][0], tour_nodes, opt_tour_len=b['tour_len'][0])
        self.best_state = copy.deepcopy(self.state)
        self.cur_step = 0
        self.done = False


    def get_state(self):
        if self.ret_best_state:
            return (self.state, self.best_state)
        return self.state

    def step(self, action):
        if self.done:
            # TODO: raise exception here??
            return self.state, 0, self.done

        self.cur_step += 1
        if action['terminate']:
            self.done = True

        if self.cur_step == self.max_num_steps:
            self.done = True

        # TODO: work on reward shaping, experiment with other reward schemes
        if self.done:
            # TODO: make a terminal state??
            # if the action given in t-1 was terminate, or cur_step == T
            # then the reward is cost(S[t-1])
            if self.ret_log_tour_len:
                reward = -np.log(self.best_state.tour_len)
            else:
                reward = -self.best_state.tour_len
        else:
            e0, e1 = action['e0'], action['e1']
            self.state.apply_move(e0, e1)
            reward = 0.
            if self.state.tour_len < self.best_state.tour_len:
                self.best_state = copy.deepcopy(self.state)

        return self.get_state(), reward, self.done

    def render(self, mode="human"):
        f = plt.figure(figsize=(8, 8))
        a = f.add_subplot(111)
        plot_utils.plot_tsp(
            a,
            self.state.nodes_coord,
            np.ones(shape=(self.state.num_nodes, self.state.num_nodes)),
            self.state.edge_weights,
            self.state.tour_adj,
            title=f'tour length = {self.state.tour_len:.3f}, opt gap = {self.state.tour_len / self.state.opt_tour_len - 1.:.3f}'
        )
        f.canvas.draw()
        img = np.frombuffer(f.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(f.canvas.get_width_height()[::-1] + (3,))
        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("TSP", img)
            cv2.waitKey(200)

        elif mode == "rgb_array":
            return img

    def close(self):
        cv2.destroyAllWindows()
