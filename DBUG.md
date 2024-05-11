# Big changes that need to incrementally tested:
- enumerate_all_tours_edges method (can be tested indepedently, API identical)
- enumerate_tour_edges method (can be tested indepedently, API identical)
- caching of tour_adj (can be tested indepedently, API identical)
- moving the processing of nbh's into envs (first moves and second)
- vectorizing of nbhs types (and reconstruction)
- vectorizing of states (and associated changes)
- grouping of multiple envs into each proc