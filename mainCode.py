import search
from datetime import datetime

t = datetime.now()
time = int(t.strftime("%H"))
trafficType = ['heavy', 'mixed', 'light']


def arrivalTime(dist, vel, trafficVol):
    kph = vel * 1.609
    opmltrvltime = dist / kph
    if (time >= 7 and time <= 9) or (time >= 15 and time <= 18):
        if trafficVol == 'heavy':
            return opmltrvltime * 1.5
        elif trafficVol == 'mixed':
            return opmltrvltime * 1.26
        else:
            return opmltrvltime
    else:
        return opmltrvltime


locs = {'SanJuan': (2044.22708, 805.68513), 'Ponce': (1993.03845, 752.62554), 'Salinas': (1989.77668, 786.16885),
        'Adjuntas': (2009.7660, 740.86058), 'Yauco': (1995.37604, 727.61380), 'Maricao': (2010.5420, 717.3230),
        'Rincon': (2028.71519, 684.92270), 'Aguadilla': (2038.46991, 694.96594), 'Camuy': (2041.50870, 726.68274),
        'Utuado': (2021.09065, 743.12228), 'Arecibo': (2040.95274, 748.58215), 'Bayamon': (2035.60690, 799.51964),
        'Catano': (2041.98468, 802.56411), 'Cayey': (2004.87415, 799.92419), 'Caguas': (2019.14630, 813.54432),
        'Juncos': (2017.81584, 191.06579), 'Canovanas': (2034.10634, 193.57450),
        'Humacao': (2009.04663, 200.84064), 'Yabucoa': (1998.14964, 195.16911), 'Guayama': (1990.80829, 805.67548),
        'Fajardo': (2028.27161, 219.65135), 'Mayaguez': (2013.45329, 696.16194)}
PR_map = search.Map({('Ponce', 'Salinas'): arrivalTime(36, 50, trafficType[0]),
                     ('Ponce', 'Adjuntas'): arrivalTime(25, 35, trafficType[2]),
                     ('Ponce', 'Yauco'): arrivalTime(32, 55, trafficType[0]),
                     ('Mayaguez', 'Yauco'): arrivalTime(45, 60, trafficType[0]),
                     ('Mayaguez', 'Maricao'): arrivalTime(26, 30, trafficType[1]),
                     ('Mayaguez', 'Rincon'): arrivalTime(23, 50, trafficType[0]),
                     ('Mayaguez', 'Aguadilla'): arrivalTime(28, 55, trafficType[0]),
                     ('Rincon', 'Aguadilla'): arrivalTime(17, 55, trafficType[0]),
                     ('Camuy', 'Aguadilla'): arrivalTime(39, 45, trafficType[0]),
                     ('Camuy', 'Utuado'): arrivalTime(44, 35, trafficType[2]),
                     ('Camuy', 'Arecibo'): arrivalTime(15, 50, trafficType[0]),
                     ('Utuado', 'Arecibo'): arrivalTime(33, 50, trafficType[1]),
                     ('Utuado', 'Adjuntas'): arrivalTime(21, 30, trafficType[2]),
                     ('Adjuntas', 'Maricao'): arrivalTime(54, 25, trafficType[2]),
                     ('Bayamon', 'Arecibo'): arrivalTime(77, 50, trafficType[0]),
                     ('Bayamon', 'Catano'): arrivalTime(11, 45, trafficType[1]),
                     ('Bayamon', 'SanJuan'): arrivalTime(19, 55, trafficType[0]),
                     ('SanJuan', 'Cayey'): arrivalTime(55, 55, trafficType[0]),
                     ('SanJuan', 'Caguas'): arrivalTime(33, 55, trafficType[0]),
                     ('SanJuan', 'Canovanas'): arrivalTime(30, 40, trafficType[1]),
                     ('Cayey', 'Salinas'): arrivalTime(32, 50, trafficType[1]),
                     ('Cayey', 'Guayama'): arrivalTime(42, 55, trafficType[1]),
                     ('Cayey', 'Caguas'): arrivalTime(24, 55, trafficType[0]),
                     ('Caguas', 'Juncos'): arrivalTime(15, 40, trafficType[1]),
                     ('Canovanas', 'Juncos'): arrivalTime(21, 50, trafficType[1]),
                     ('Canovanas', 'Fajardo'): arrivalTime(31, 50, trafficType[0]),
                     ('Humacao', 'Fajardo'): arrivalTime(36, 55, trafficType[0]),
                     ('Humacao', 'Yabucoa'): arrivalTime(18, 55, trafficType[1]),
                     ('Guayama', 'Salinas'): arrivalTime(23, 50, trafficType[0]),
                     ('Guayama', 'Yabucoa'): arrivalTime(38, 40, trafficType[1])}, locs)

PR_vel = search.Map({('Ponce', 'Salinas'): 50, ('Ponce', 'Adjuntas'): 40,
                     ('Ponce', 'Yauco'): 55,
                     ('Mayaguez', 'Yauco'): 55, ('Mayaguez', 'Maricao'): 40,
                     ('Mayaguez', 'Rincon'): 50,
                     ('Mayaguez', 'Aguadilla'): 55,
                     ('Rincon', 'Aguadilla'): 55,
                     ('Camuy', 'Aguadilla'): 50, ('Camuy', 'Utuado'): 40,
                     ('Camuy', 'Arecibo'): 50,
                     ('Utuado', 'Arecibo'): 50, ('Utuado', 'Adjuntas'): 35,
                     ('Adjuntas', 'Maricao'): 35,
                     ('Bayamon', 'Arecibo'): 50, ('Bayamon', 'Catano'): 45,
                     ('Bayamon', 'SanJuan'): 55,
                     ('SanJuan', 'Cayey'): 55, ('SanJuan', 'Caguas'): 55,
                     ('SanJuan', 'Canovanas'): 50,
                     ('Cayey', 'Salinas'): 50, ('Cayey', 'Guayama'): 55,
                     ('Cayey', 'Caguas'): 55,
                     ('Caguas', 'Juncos'): 50,
                     ('Canovanas', 'Juncos'): 50,
                     ('Canovanas', 'Fajardo'): 50,
                     ('Humacao', 'Fajardo'): 55, ('Humacao', 'Yabucoa'): 55,
                     ('Guayama', 'Salinas'): 50,
                     ('Guayama', 'Yabucoa'): 50}, locs)

# locs= 'City':(UTM_Northing, UTM_Easting)
g = search.RouteProblem('Mayaguez', "SanJuan", map=PR_map)
g1 = search.RouteProblem('Canovanas', "Arecibo", map=PR_map)
g2 = search.RouteProblem('Caguas', "Yauco", map=PR_map)
g3 = search.RouteProblem('Ponce', "Rincon", map=PR_map)
g4 = search.RouteProblem('Yabucoa', "Maricao", map=PR_map)
g5 = search.RouteProblem('Utuado', "Cayey", map=PR_map)

print("////////////g////////////////")
bfs = search.breadth_first_search(g)
dfs = search.depth_first_recursive_search(g)
ucs = search.uniform_cost_search(g)
ids = search.iterative_deepening_search(g)
gs = search.greedy_bfs(g)
bidi = search.bidirectional_uniform_cost_search(g)
print("This is bfs ", bfs.path_cost)
print("This is dfs ", dfs.path_cost)
print("This is ucs ", ucs.path_cost)
print("This is ids ", ids.path_cost)
print("This is greedy ", gs.path_cost)
print("This is bidirectional search ", bidi.path_cost)
# print("Reporting", search.report((search.uniform_cost_search, search.iterative_deepening_search,
#                                   search.depth_first_recursive_search, search.bidirectional_uniform_cost_search
#                                   , search.greedy_bfs, search.breadth_first_search), [g]))
print("////////////g1////////////////")
bfs = search.breadth_first_search(g1)
dfs = search.depth_first_recursive_search(g1)
ucs = search.uniform_cost_search(g1)
ids = search.iterative_deepening_search(g1)
gs = search.greedy_bfs(g1)
bidi = search.bidirectional_uniform_cost_search(g1)
print("This is bfs ", bfs.path_cost)
print("This is dfs ", dfs.path_cost)
print("This is ucs ", ucs.path_cost)
print("This is ids ", ids.path_cost)
print("This is greedy ", gs.path_cost)
print("This is bidirectional search ", bidi.path_cost)
# print("Reporting", search.report((search.uniform_cost_search, search.iterative_deepening_search,
#                                   search.depth_first_recursive_search, search.bidirectional_uniform_cost_search
#                                   , search.greedy_bfs, search.breadth_first_search), [g1]))
print("////////////g2////////////////")
bfs = search.breadth_first_search(g2)
dfs = search.depth_first_recursive_search(g2)
ucs = search.uniform_cost_search(g2)
ids = search.iterative_deepening_search(g2)
gs = search.greedy_bfs(g2)
bidi = search.bidirectional_uniform_cost_search(g2)
print("This is bfs ", bfs.path_cost)
print("This is dfs ", dfs.path_cost)
print("This is ucs ", ucs.path_cost)
print("This is ids ", ids.path_cost)
print("This is greedy ", gs.path_cost)
print("This is bidirectional search ", bidi.path_cost)
# print("Reporting", search.report((search.uniform_cost_search, search.iterative_deepening_search,
#                                   search.depth_first_recursive_search, search.bidirectional_uniform_cost_search
#                                   , search.greedy_bfs, search.breadth_first_search), [g2]))
print("////////////g3////////////////")
bfs = search.breadth_first_search(g3)
dfs = search.depth_first_recursive_search(g3)
ucs = search.uniform_cost_search(g3)
ids = search.iterative_deepening_search(g3)
gs = search.greedy_bfs(g3)
bidi = search.bidirectional_uniform_cost_search(g3)
print("This is bfs ", bfs.path_cost, search.path_states(bfs))
print("This is dfs ", dfs.path_cost, search.path_states(dfs))
print("This is ucs ", ucs.path_cost, search.path_states(ucs))
print("This is ids ", ids.path_cost, search.path_states(ids))
print("This is greedy ", gs.path_cost, search.path_states(gs))
print("This is bidirectional search ", bidi.path_cost, search.path_states(bidi))
# print("Reporting", search.report((search.uniform_cost_search, search.iterative_deepening_search,
#                                   search.depth_first_recursive_search, search.bidirectional_uniform_cost_search
#                                   , search.greedy_bfs, search.breadth_first_search), [g3]))
# print("////////////g4////////////////")
bfs = search.breadth_first_search(g4)
dfs = search.depth_first_recursive_search(g4)
ucs = search.uniform_cost_search(g4)
ids = search.iterative_deepening_search(g4)
gs = search.greedy_bfs(g4)
bidi = search.bidirectional_uniform_cost_search(g4)
print("This is bfs ", bfs.path_cost)
print("This is dfs ", dfs.path_cost)
print("This is ucs ", ucs.path_cost)
print("This is ids ", ids.path_cost)
print("This is greedy ", gs.path_cost)
print("This is bidirectional search ", bidi.path_cost)
# print("Reporting", search.report((search.uniform_cost_search, search.iterative_deepening_search,
#                                   search.depth_first_recursive_search, search.bidirectional_uniform_cost_search
#                                   , search.greedy_bfs, search.breadth_first_search), [g4]))
print("////////////g5////////////////")
bfs = search.breadth_first_search(g5)
dfs = search.depth_first_recursive_search(g5)
ucs = search.uniform_cost_search(g5)
ids = search.iterative_deepening_search(g5)
gs = search.greedy_bfs(g5)
bidi = search.bidirectional_uniform_cost_search(g5)
print("This is bfs ", bfs.path_cost)
print("This is dfs ", dfs.path_cost)
print("This is ucs ", ucs.path_cost)
print("This is ids ", ids.path_cost)
print("This is greedy ", gs.path_cost)
print("This is bidirectional search ", bidi.path_cost)
# print("Reporting", search.report((search.uniform_cost_search, search.iterative_deepening_search,
#                                   search.depth_first_recursive_search, search.bidirectional_uniform_cost_search
#                                   , search.greedy_bfs, search.breadth_first_search), [g5]))
print("Reporting", search.report((search.uniform_cost_search, search.iterative_deepening_search,
                                  search.depth_first_recursive_search, search.bidirectional_uniform_cost_search
                                  , search.greedy_bfs, search.breadth_first_search), [g,g1,g2,g3,g4,g5]))