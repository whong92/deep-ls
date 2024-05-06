# Big changes that need to incrementally tested:
- enumerate_all_tours_edges method (can be tested indepedently, API identical)
- enumerate_tour_edges method (can be tested indepedently, API identical)
- caching of tour_adj (can be tested indepedently, API identical)
- moving the processing of nbh's into envs (first moves and second)
- vectorizing of nbhs types (and reconstruction)  --  needs to be done after moving processing into nbh's?
-- can be done by still passing states together with vectorized state.
- grouping of multiple envs into each proc