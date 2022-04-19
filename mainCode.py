import search

romania_map = search.UndirectedGraph(dict(
    Ponce=dict(Salinas=36, Adjuntas=25, Yauco=32),
    Yauco=dict(Ponce=32, Mayaguez=45),
    Mayaguez=dict(Yauco=45, Maricao=26, Rincon=23, Aguadilla=28),
    Maricao=dict(Mayaguez=26, Adjuntas=54),
    Adjuntas=dict(Maricao=54, Ponce=25, Utuado=21),
    Rincon=dict(Mayaguez=23, Aguadilla=17),
    Aguadilla=dict(Rincon=17, Mayaguez=28, Camuy=39),
    Camuy=dict(Aguadilla=39, Arecibo=15, Utuado=44),
    Utuado=dict(Camuy=44, Arecibo=33, Adjuntas=21),
    Arecibo=dict(Camuy=15, Utuado=33, Bayamon=77),
    Bayamon=dict(Arecibo=77, Catano=11, SanJuan=19),
    Catano=dict(Bayamon=11),
    SanJuan=dict(Bayamon=19, Cayey=55, Caguas=33, Canovanas=30),
    Cayey=dict(SanJuan=55, Salinas=32, Guayama=42, Caguas=24),
    Caguas=dict(SanJuan=33, Cayey=24, Juncos=15),
    Canovanas=dict(SanJuan=30, Juncos=22, Fajardo=31),
    Juncos=dict(Caguas=15, Canovanas=22, Fajardo=52),
    Fajardo=dict(Juncos=52, Humacao=36, Canovanas=31),
    Salinas=dict(Cayey=32, Ponce=36, Guayama=23),
    Humacao=dict(Fajardo=36, Yabucoa=18),
    Guayama=dict(Yabucoa=38, Cayey=42, Salinas=23)))
#Velocidades
# Ponce a Yauco = 65mph Ponce a Isablea = 55
# Mayaguez a Yauco = 65mph    Mayaguez a Rincon = 60  Mayaguez a Aguadilla = 65 Mayaguez a Maricao =35
# 


# romania_map.locations = dict(
#     Arad=(91, 492), Bucharest=(400, 327), Craiova=(253, 288),
#     Drobeta=(165, 299), Eforie=(562, 293), Fagaras=(305, 449),
#     Giurgiu=(375, 270), Hirsova=(534, 350), Iasi=(473, 506),
#     Lugoj=(165, 379), Mehadia=(168, 339), Neamt=(406, 537),
#     Oradea=(131, 571), Pitesti=(320, 368), Rimnicu=(233, 410),
#     Sibiu=(207, 457), Timisoara=(94, 410), Urziceni=(456, 350),
#     Vaslui=(509, 444), Zerind=(108, 531))
g = search.GraphProblem('Mayaguez', "SanJuan", romania_map)
p1 = search.breadth_first_graph_search(g)
print("This is breadth first search", p1)
p2 = search.best_first_graph_search(g, lambda node: node.path_cost)
print("This is uniform cost search", p2)
p3 = search.depth_first_graph_search(g)
print("This is depth cost search", p3)
p4 = search.greedy_best_first_graph_search(g, lambda n: g.h(n))
print("This is greedy search", p4)
p5 = search.astar_search(g, lambda n: g.h(n) + n.path_cost)
print("This is a* search", p5)
