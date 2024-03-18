#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <algorithm>    // std::set_difference, std::sort
#include <vector>
#include <tuple>
#include <set>
#include <unordered_map>
#include <string>


using namespace std;

//it is only a trick to ensure import_array() is called, when *.so is loaded
//just called only once
int init_numpy(void){
    import_array(); // PyError if not successful
    return 0;
}

const static int numpy_initialized =  init_numpy();

int add(int a, int b) {
    return a * b;
}


void print_arr(PyArrayObject* arr) {
    int N = PyArray_Size((PyObject*) arr);
    printf("[");
    for (int i=0; i < N; i++){
        int a = *((int*) PyArray_GETPTR1(arr, i));
        printf("%d, ", a);
    }
    printf("]\n");
}


tuple<int, int> make_normalized_edge(int u, int v){
    if (u == -1){
        u = 0;
    }
    if (v == -1){
        v = 0;
    }
    if (u > v) {
        return tuple<int, int>{v, u};
    }
    return tuple<int, int>{u, v};
}


vector<tuple<int, int>> get_edge_difference(
    set<tuple<int, int>> src_edges,
    set<tuple<int, int>> dst_edges
) {
    vector<tuple<int, int>> edges_diff(src_edges.size());
    vector<tuple<int, int>>::iterator it;
    it=set_difference(
        src_edges.begin(),
        src_edges.end(),
        dst_edges.begin(),
        dst_edges.end(),
        edges_diff.begin()
    );
    edges_diff.resize(it - edges_diff.begin());
    sort(edges_diff.begin(), edges_diff.end());
    return edges_diff;
}


vector<tuple<int, int>> get_edge_sym_difference(
    set<tuple<int, int>> src_edges,
    set<tuple<int, int>> dst_edges
) {
    vector<tuple<int, int>> edges_diff;
    set_symmetric_difference(
        src_edges.begin(),
        src_edges.end(),
        dst_edges.begin(),
        dst_edges.end(),
        back_inserter(edges_diff)
    );
    sort(edges_diff.begin(), edges_diff.end());
    return edges_diff;
}


void print_set_edges(set<tuple<int, int>> edges) {
    set<tuple<int, int>>::iterator it;
    printf("[");
    for (it = edges.begin(); it != edges.end(); it++){
        printf("(%d, %d), ", get<0>(*it), get<1>(*it));
    }
    printf("]\n");
}


void print_vector_edges(vector<tuple<int, int>> edges) {
    vector<tuple<int, int>>::iterator it;
    printf("[");
    for (it = edges.begin(); it != edges.end(); it++){
        printf("(%d, %d), ", get<0>(*it), get<1>(*it));
    }
    printf("]\n");
}


void make_and_insert_normalized_edge(int u, int v, set<tuple<int, int>>* edges) {
    tuple<int, int> e = make_normalized_edge(u, v);
    // printf("make_and_insert_normalized_edge %d, %d -> %d, %d equals %d\n", u, v, get<0>(e), get<1>(e), get<0>(e) != get<1>(e));
    if (get<0>(e) != get<1>(e)) edges->insert(e);
}


struct normalized_nbh {
    set<tuple<int, int>> src_edges;
    set<tuple<int, int>> dst_edges;
    vector<tuple<int, int>> edges_added;
    vector<tuple<int, int>> edges_removed;
    vector<tuple<int, int>> edges_diff;
};


double get_cost(normalized_nbh nbh, PyArrayObject* edge_weights) {
    double cost = 0.;
    for(int i=0; i < nbh.edges_added.size(); i++){
       tuple<int, int> e = nbh.edges_added[i];
       cost -= *((double*) PyArray_GETPTR2(edge_weights, get<0>(e),  get<1>(e)));
    }
    for(int i=0; i < nbh.edges_removed.size(); i++){
       tuple<int, int> e = nbh.edges_removed[i];
       cost += *((double*) PyArray_GETPTR2(edge_weights, get<0>(e),  get<1>(e)));
    }
    return cost;
}


tuple<int, int> get_edge_pair_at_offset(PyArrayObject* edge_pairs, int row, int col) {
    int e0_n0 = *((int*) PyArray_GETPTR2(edge_pairs, row, col));
    int e0_n1 = *((int*) PyArray_GETPTR2(edge_pairs, row, col + 1));
    return {e0_n0, e0_n1};
}


bool check_relocate_move_valid(
    int src_node, PyObject* cum_demand_dst, PyArrayObject* demands, double max_tour_dem
) {
    double dst_demand = PyFloat_AsDouble(PyDict_GetItem(cum_demand_dst,  Py_BuildValue("i", -1)));
    double src_demand = *((double*) PyArray_GETPTR1(demands, src_node - 1));
    return (dst_demand + src_demand) <= max_tour_dem;
}


string get_nbh_rep_key(vector<tuple<int, int>> edges_added, vector<tuple<int, int>> edges_removed) {
    vector<tuple<int, int>>::iterator it;
    string nbh_rep_key = "[";
    for (it = edges_added.begin(); it != edges_added.end(); it++){
        nbh_rep_key.append("(");
        nbh_rep_key.append(to_string(get<0>(*it)));
        nbh_rep_key.append(",");
        nbh_rep_key.append(to_string(get<1>(*it)));
        nbh_rep_key.append(")");
    }
    nbh_rep_key.append("],[");
    for (it = edges_removed.begin(); it != edges_removed.end(); it++){
        nbh_rep_key.append("(");
        nbh_rep_key.append(to_string(get<0>(*it)));
        nbh_rep_key.append(",");
        nbh_rep_key.append(to_string(get<1>(*it)));
        nbh_rep_key.append(")");
    }
    nbh_rep_key.append("]");
    return nbh_rep_key;
}


PyObject* map_to_pydict(unordered_map<string, string> map) {
    PyObject* py_dict = PyDict_New();

    unordered_map<string, string>::iterator it;
    for (it = map.begin(); it != map.end(); it++)
    {
        string k = it->first;
        string v = it->second;
        PyDict_SetItem(py_dict, Py_BuildValue("s", k.c_str()), Py_BuildValue("s", v.c_str()));
    }
    return py_dict;
}


PyObject* to_py_tuple(tuple<int, int> tup) {
    return Py_BuildValue("(ii)", get<0>(tup), get<1>(tup));
}


PyObject*  parse_args_flatten_reloc(
    PyObject* reloc_nbhs,
    PyObject* tours_dict,
    PyArrayObject* node_demands,
    PyArrayObject* edge_weights,
    double max_tour_demand
) {
    Py_ssize_t num_reloc_nbhs = PyList_Size(reloc_nbhs);
    PyObject* reloc_nbhs_dict = PyDict_New();
    for (int i=0; i<num_reloc_nbhs; i++) {
        PyObject* reloc_nbh = PyList_GetItem(reloc_nbhs, i);
        PyObject* tour0 = PyDict_GetItemString(reloc_nbh,  "src_tour");
        PyObject* tour1 = PyDict_GetItemString(reloc_nbh,  "dst_tour");
        PyArrayObject* node_edge_pairs = (PyArrayObject*) PyDict_GetItemString(reloc_nbh,  "node_edges_pairs");
        int n_pairs = PyArray_DIM(node_edge_pairs, 0);

        for (int j=0; j<n_pairs; j++) {
            int src_node = *((int*) PyArray_GETPTR2(node_edge_pairs, j, 0));
            tuple<int, int> src_u = get_edge_pair_at_offset(node_edge_pairs, j, 1);
            tuple<int, int> src_v = get_edge_pair_at_offset(node_edge_pairs, j, 3);
            tuple<int, int> dst_w = get_edge_pair_at_offset(node_edge_pairs, j, 5);

            set<tuple<int, int>> src_edges;
            make_and_insert_normalized_edge(get<0>(src_u), get<1>(src_u), &src_edges);
            make_and_insert_normalized_edge(get<0>(src_v), get<1>(src_v), &src_edges);
            make_and_insert_normalized_edge(get<0>(dst_w), get<1>(dst_w), &src_edges);

            set<tuple<int, int>> dst_edges;
            make_and_insert_normalized_edge(get<0>(src_u), get<1>(src_v), &dst_edges);
            make_and_insert_normalized_edge(get<0>(dst_w), src_node, &dst_edges);
            make_and_insert_normalized_edge(src_node, get<1>(dst_w), &dst_edges);

            vector<tuple<int, int>> edges_added = get_edge_difference(src_edges, dst_edges);
            vector<tuple<int, int>> edges_removed = get_edge_difference(dst_edges, src_edges);


            vector<tuple<int, int>> edges_diff = get_edge_sym_difference(src_edges, dst_edges);
            normalized_nbh normalized_nbh_reloc = {
                .src_edges = src_edges,
                .dst_edges = dst_edges,
                .edges_added = edges_added,
                .edges_removed = edges_removed,
                .edges_diff = edges_diff
            };
            bool noop = edges_diff.size() == 0;
            bool move_valid = true;
            if (!noop) {
                if(tour1 != Py_None) {
                    PyObject* dst_tour = PyDict_GetItem(tours_dict,  tour1);
                    PyObject* dst_tour_cum_dems = PyDict_GetItemString(dst_tour,  "cum_dems");
                    move_valid = check_relocate_move_valid(
                        src_node,
                        dst_tour_cum_dems,
                        node_demands,
                        max_tour_demand
                    );
                }
                if (move_valid) {
                    double cost = get_cost(
                        normalized_nbh_reloc, edge_weights
                    );
//                    reloc_nbhs_dict[
//                        get_nbh_rep_key(
//                            normalized_nbh_reloc.edges_added,
//                            normalized_nbh_reloc.edges_removed
//                        )
//                    ] = to_string(cost);

                    PyObject* nbh_dict = PyDict_New();
                    PyDict_SetItemString(nbh_dict, "nb_type", Py_BuildValue("s", "reloc"));
                    PyDict_SetItemString(nbh_dict, "tour0", tour0);
                    if(tour1 != Py_None) { PyDict_SetItemString(nbh_dict, "tour1", tour1); }
                    else { PyDict_SetItemString(nbh_dict, "tour1", Py_BuildValue("i", -1)); }
                    PyDict_SetItemString(nbh_dict, "src_node", Py_BuildValue("i", src_node));
                    PyDict_SetItemString(nbh_dict, "src_u", to_py_tuple(src_u));
                    PyDict_SetItemString(nbh_dict, "src_v", to_py_tuple(src_v));
                    PyDict_SetItemString(nbh_dict, "dst_w", to_py_tuple(dst_w));
                    PyDict_SetItemString(nbh_dict, "dst_wp", Py_BuildValue("(i,i)", get<0>(src_u), get<1>(src_v)));
                    PyDict_SetItemString(nbh_dict, "src_up", Py_BuildValue("(i,i)", get<0>(dst_w), src_node));
                    PyDict_SetItemString(nbh_dict, "src_vp", Py_BuildValue("(i,i)", src_node, get<1>(dst_w)));
                    PyDict_SetItemString(nbh_dict, "cost", Py_BuildValue("d", cost));
                    PyDict_SetItemString(
                        reloc_nbhs_dict,
                        get_nbh_rep_key(normalized_nbh_reloc.edges_added, normalized_nbh_reloc.edges_removed).c_str(),
                        nbh_dict
                    );

                }
            }

        }
    }

    return reloc_nbhs_dict;
}

static PyObject* py_flatten_reloc_nbh(PyObject* self, PyObject* args) {
    PyObject* tours_dict;
    PyObject* reloc_nbhs;
    PyObject* node_demands;
    PyObject* edge_weights;
    double max_tour_demands;
    if (!PyArg_ParseTuple(args, "OOOOd", &reloc_nbhs, &tours_dict, &node_demands, &edge_weights, &max_tour_demands))
        return NULL;
    if (!PyDict_Check(tours_dict)){
        PyErr_SetString(PyExc_TypeError, "Expected a dict");
        return NULL;
    }
    if (!PyList_Check(reloc_nbhs)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list.");
        return NULL;
    }
    return parse_args_flatten_reloc(
        reloc_nbhs,
        tours_dict,
        (PyArrayObject* ) node_demands,
        (PyArrayObject* ) edge_weights,
        max_tour_demands
    );
}


bool check_cross_move_valid(
    tuple<int, int> e0,
    tuple<int, int> e1,
    tuple<int, int> e0p,
    tuple<int, int> e1p,
    PyObject* cum_demand_0,
    PyObject* cum_demand_1,
    double max_tour_dem
) {
    int u = get<0>(e0);
    int v = get<1>(e0);
    int x = get<0>(e1);
    int y = get<1>(e1);

    double cum_dem_u = PyFloat_AsDouble(PyDict_GetItem(cum_demand_0,  Py_BuildValue("i", u)));
    double cum_dem_v = PyFloat_AsDouble(PyDict_GetItem(cum_demand_0,  Py_BuildValue("i", v)));
    double cum_dem_x = PyFloat_AsDouble(PyDict_GetItem(cum_demand_1,  Py_BuildValue("i", x)));
    double cum_dem_y = PyFloat_AsDouble(PyDict_GetItem(cum_demand_1,  Py_BuildValue("i", y)));

    double C0 = PyFloat_AsDouble(PyDict_GetItem(cum_demand_0,  Py_BuildValue("i", -1)));
    double C1 = PyFloat_AsDouble(PyDict_GetItem(cum_demand_1,  Py_BuildValue("i", -1)));

    // sanity check for the ordering since u places before v in the tour
    assert(cum_dem_u <= cum_dem_v);
    assert(cum_dem_x <= cum_dem_y);

    double tour_0_new_dem = -1;
    double tour_1_new_dem = -1;
    // case ux vy
    if (get<1>(e0p) == x && get<1>(e1p) == y) {
        // 0 -> u -> x -> 0
        tour_0_new_dem = cum_dem_u + cum_dem_x;
        // -1 -> v -> y -> -1
        tour_1_new_dem = (C0 - cum_dem_u) + (C1 - cum_dem_x);
    // case uy vx
    } else if (get<1>(e0p) == y && get<1>(e1p) == x) {
        // 0 -> u -> y -> -1
        tour_0_new_dem = cum_dem_u + (C1 - cum_dem_x);
        // 0 -> x -> v -> -1
        tour_1_new_dem = cum_dem_x + (C0 - cum_dem_u);
    } else {
        assert(false);
    }

    return (tour_0_new_dem <= max_tour_dem) && (tour_1_new_dem <= max_tour_dem);
}


normalized_nbh get_normalized_nbh(
    tuple<int, int> e0,
    tuple<int, int> e1,
    tuple<int, int> e0p,
    tuple<int, int> e1p
) {
    set<tuple<int, int>> src_edges;
    make_and_insert_normalized_edge(get<0>(e0), get<1>(e0), &src_edges);
    make_and_insert_normalized_edge(get<0>(e1), get<1>(e1), &src_edges);

    set<tuple<int, int>> dst_edges;
    make_and_insert_normalized_edge(get<0>(e0p), get<1>(e0p), &dst_edges);
    make_and_insert_normalized_edge(get<0>(e1p), get<1>(e1p), &dst_edges);

    vector<tuple<int, int>> edges_added = get_edge_difference(src_edges, dst_edges);
    vector<tuple<int, int>> edges_removed = get_edge_difference(dst_edges, src_edges);
    vector<tuple<int, int>> edges_diff = get_edge_sym_difference(src_edges, dst_edges);

    return normalized_nbh {
        .src_edges = src_edges,
        .dst_edges = dst_edges,
        .edges_added = edges_added,
        .edges_removed = edges_removed,
        .edges_diff = edges_diff
    };

}


PyObject* parse_args_flatten_cross(
    PyObject* cross_nbhs,
    PyObject* tours_dict,
    PyArrayObject* node_demands,
    double max_tour_demand,
    PyArrayObject* edge_weights
) {

    Py_ssize_t size = PyList_Size(cross_nbhs);

    PyObject* cross_nbhs_dict = PyDict_New();
    for (int i=0; i<size; i++) {
        PyObject* cross_nbh = PyList_GetItem(cross_nbhs, i);
        PyObject* tour0 = PyDict_GetItemString(cross_nbh,  "src_tour");
        PyObject* tour1 = PyDict_GetItemString(cross_nbh,  "dst_tour");

        PyArrayObject* tour_edge_pairs = (PyArrayObject*) PyDict_GetItemString(cross_nbh,  "tour_edges_pairs");
        int n_pairs = PyArray_DIM(tour_edge_pairs, 0);

        for (int j=0; j<n_pairs; j++) {
            PyObject* src_tour = PyDict_GetItem(tours_dict,  tour0);
            PyObject* dst_tour = PyDict_GetItem(tours_dict,  tour1);
            PyObject* src_tour_cum_dems = PyDict_GetItemString(src_tour,  "cum_dems");
            PyObject* dst_tour_cum_dems = PyDict_GetItemString(dst_tour,  "cum_dems");

            tuple<int, int> e0 = get_edge_pair_at_offset(tour_edge_pairs, j, 0);
            tuple<int, int> e1 = get_edge_pair_at_offset(tour_edge_pairs, j, 2);

            // ================= first move
            tuple<int, int> e0p0 = get_edge_pair_at_offset(tour_edge_pairs, j, 4);
            tuple<int, int> e1p0 = get_edge_pair_at_offset(tour_edge_pairs, j, 6);

            normalized_nbh normalized_nbh_0 = get_normalized_nbh(e0, e1, e0p0, e1p0);
            double cost0 = get_cost(normalized_nbh_0, edge_weights);

            bool noop = normalized_nbh_0.edges_diff.size() == 0;

            if(!noop) {
                bool valid = check_cross_move_valid(
                    e0, e1, e0p0, e1p0, src_tour_cum_dems, dst_tour_cum_dems, max_tour_demand
                );
                if(valid) {
                    PyObject* nbh_dict = PyDict_New();
                    PyDict_SetItemString(nbh_dict, "nb_type", Py_BuildValue("s", "cross"));
                    PyDict_SetItemString(nbh_dict, "tour0", tour0);
                    PyDict_SetItemString(nbh_dict, "tour1", tour1);
                    PyDict_SetItemString(nbh_dict, "e0", to_py_tuple(e0));
                    PyDict_SetItemString(nbh_dict, "e1", to_py_tuple(e1));
                    PyDict_SetItemString(nbh_dict, "e0p", to_py_tuple(e0p0));
                    PyDict_SetItemString(nbh_dict, "e1p", to_py_tuple(e1p0));
                    PyDict_SetItemString(nbh_dict, "cost", Py_BuildValue("d", cost0));
                    PyDict_SetItemString(
                        cross_nbhs_dict,
                        get_nbh_rep_key(normalized_nbh_0.edges_added, normalized_nbh_0.edges_removed).c_str(),
                        nbh_dict
                    );
                }
            }

            // ================= second move
            tuple<int, int> e0p1 = get_edge_pair_at_offset(tour_edge_pairs, j, 8);
            tuple<int, int> e1p1 = get_edge_pair_at_offset(tour_edge_pairs, j, 10);

            normalized_nbh normalized_nbh_1 = get_normalized_nbh(e0, e1, e0p1, e1p1);
            double cost1 = get_cost(normalized_nbh_1, edge_weights);
            noop = normalized_nbh_1.edges_diff.size() == 0;

            if(!noop) {
                bool valid = check_cross_move_valid(
                    e0, e1, e0p1, e1p1, src_tour_cum_dems, dst_tour_cum_dems, max_tour_demand
                );
                if(valid) {
                    // cross_nbhs_dict[get_nbh_rep_key(normalized_nbh_1.edges_added, normalized_nbh_1.edges_removed)] = to_string(cost1);
                    PyObject* nbh_dict = PyDict_New();
                    PyDict_SetItemString(nbh_dict, "nb_type", Py_BuildValue("s", "cross"));
                    PyDict_SetItemString(nbh_dict, "tour0", tour0);
                    PyDict_SetItemString(nbh_dict, "tour1", tour1);
                    PyDict_SetItemString(nbh_dict, "e0", to_py_tuple(e0));
                    PyDict_SetItemString(nbh_dict, "e1", to_py_tuple(e1));
                    PyDict_SetItemString(nbh_dict, "e0p", to_py_tuple(e0p1));
                    PyDict_SetItemString(nbh_dict, "e1p", to_py_tuple(e1p1));
                    PyDict_SetItemString(nbh_dict, "cost", Py_BuildValue("d", cost1));
                    PyDict_SetItemString(
                        cross_nbhs_dict,
                        get_nbh_rep_key(normalized_nbh_1.edges_added, normalized_nbh_1.edges_removed).c_str(),
                        nbh_dict
                    );
                }
            }
        }

    }

    return cross_nbhs_dict;
}

PyObject* parse_args_flatten_2opt(
    PyObject* twoopt_nbhs,
    PyObject* tours_dict,
    PyArrayObject* edge_weights
) {
    Py_ssize_t n_twoopt_nbhs = PyList_Size(twoopt_nbhs);

    PyObject* twoopt_nbhs_dict = PyDict_New();
    for (int i=0; i<n_twoopt_nbhs; i++) {
        PyObject* twoopt_nbh = PyList_GetItem(twoopt_nbhs, i);
        PyObject* tour0 = PyDict_GetItemString(twoopt_nbh,  "tour_idx");
        PyArrayObject* tour_edge_pairs = (PyArrayObject*) PyDict_GetItemString(twoopt_nbh,  "tour_edges_pairs");
        int n_pairs = PyArray_DIM(tour_edge_pairs, 0);

        for (int j=0; j<n_pairs; j++) {
            tuple<int, int> e0 = get_edge_pair_at_offset(tour_edge_pairs, j, 0);
            tuple<int, int> e1 = get_edge_pair_at_offset(tour_edge_pairs, j, 2);
            tuple<int, int> e0p = get_edge_pair_at_offset(tour_edge_pairs, j, 4);
            tuple<int, int> e1p = get_edge_pair_at_offset(tour_edge_pairs, j, 6);

            normalized_nbh normalized_nbh = get_normalized_nbh(e0, e1, e0p, e1p);
            double cost = get_cost(normalized_nbh, edge_weights);

            bool noop = normalized_nbh.edges_diff.size() == 0;

            // this will be the dedup key used in
            if(!noop) {
//                twoopt_nbhs_dict[get_nbh_rep_key(normalized_nbh.edges_added, normalized_nbh.edges_removed)] = to_string(cost);

                PyObject* nbh_dict = PyDict_New();
                PyDict_SetItemString(nbh_dict, "nb_type", Py_BuildValue("s", "2opt"));
                PyDict_SetItemString(nbh_dict, "tour_idx", tour0);
                PyDict_SetItemString(nbh_dict, "e0", to_py_tuple(e0));
                PyDict_SetItemString(nbh_dict, "e1", to_py_tuple(e1));
                PyDict_SetItemString(nbh_dict, "e0p", to_py_tuple(e0p));
                PyDict_SetItemString(nbh_dict, "e1p", to_py_tuple(e1p));
                PyDict_SetItemString(nbh_dict, "cost", Py_BuildValue("d", cost));
                PyDict_SetItemString(
                    twoopt_nbhs_dict,
                    get_nbh_rep_key(normalized_nbh.edges_added, normalized_nbh.edges_removed).c_str(),
                    nbh_dict
                );
            }

        }
    }

//    return map_to_pydict(twoopt_nbhs_dict);
    return twoopt_nbhs_dict;
}


static PyObject* py_flatten_2opt_nbh(PyObject* self, PyObject* args) {
    PyObject* tours_dict;
    PyObject* twoopt_nbhs;
    PyObject* edge_weights;
    if (!PyArg_ParseTuple(args, "OOO", &twoopt_nbhs, &tours_dict, &edge_weights))
        return NULL;
    if (!PyDict_Check(tours_dict)){
        PyErr_SetString(PyExc_TypeError, "Expected a dict");
        return NULL;
    }
    if (!PyList_Check(twoopt_nbhs)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list.");
        return NULL;
    }
    return parse_args_flatten_2opt(twoopt_nbhs, tours_dict, (PyArrayObject* ) edge_weights);
}


static PyObject* py_flatten_cross_nbh(PyObject* self, PyObject* args) {
    PyObject* tours_dict;
    PyObject* cross_nbhs;
    PyObject* node_demands;
    PyObject* edge_weights;
    double max_tour_demands;
    if (!PyArg_ParseTuple(args, "OOOOd", &cross_nbhs, &tours_dict, &node_demands, &edge_weights, &max_tour_demands))
        return NULL;
    if (!PyDict_Check(tours_dict)){
        PyErr_SetString(PyExc_TypeError, "Expected a dict");
        return NULL;
    }
    if (!PyList_Check(cross_nbhs)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list.");
        return NULL;
    }
    return parse_args_flatten_cross(
        cross_nbhs,
        tours_dict,
        (PyArrayObject* ) node_demands,
        max_tour_demands,
        (PyArrayObject* ) edge_weights
    );
}

static PyObject* py_add(PyObject* self, PyObject* args) {
    int a, b;
    if (!PyArg_ParseTuple(args, "ii", &a, &b))
        return NULL;
    return Py_BuildValue("i", add(a, b));
}

static PyMethodDef MyMethods[] = {
    {"add",  py_add, METH_VARARGS, "Add two numbers"},
    {"flatten_2opt_nbh",  py_flatten_2opt_nbh, METH_VARARGS, "Flatten 2Opt Nbh"},
    {"flatten_reloc_nbh",  py_flatten_reloc_nbh, METH_VARARGS, "Flatten Reloc Nbh"},
    {"flatten_cross_nbh",  py_flatten_cross_nbh, METH_VARARGS, "Flatten Cross Nbh"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef vrpstate = {
    PyModuleDef_HEAD_INIT,
    "vrpstate",
    NULL,
    -1,
    MyMethods
};

PyMODINIT_FUNC PyInit_vrpstate(void) {
    return PyModule_Create(&vrpstate);
}