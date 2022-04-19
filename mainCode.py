import search
trafficType = ['heavy','mixed','light'] 

PR_map = search.Map({('Ponce','Salinas'):  [36,trafficType[0]],('Ponce','Adjuntas'):  [25,trafficType[2]],('Ponce','Yauco'):  32,
('Mayaguez','Yauco'):  [45,trafficType[0]],('Mayaguez','Maricao'):  [26,trafficType[1]],('Mayaguez','Rincon'):  [23,trafficType[0]],('Mayaguez','Aguadilla'):  [28,trafficType[0]],
('Rincon','Aguadilla'):  [17,trafficType[0]],('Camuy','Aguadilla'):  [39,trafficType[0]],('Camuy','Utuado'):  [44,trafficType[2]],('Camuy','Arecibo'):  [15,trafficType[0]],
('Utuado','Arecibo'):  [33,trafficType[1]],('Utuado','Adjuntas'):  [21,trafficType[2]],('Adjuntas','Maricao'):  [54,trafficType[2]],
('Bayamon','Arecibo'):  [77,trafficType[0]],('Bayamon','Catano'):  [11,trafficType[1]],('Bayamon','SanJuan'):  [19,trafficType[0]],
('SanJuan','Cayey'):  [55,trafficType[0]],('SanJuan','Caguas'):  [33,trafficType[0]],('SanJuan','Canovanas'):  [30,trafficType[1]],
('Cayey','Salinas'):  [32,trafficType[1]],('Cayey','Guayama'):  [42,trafficType[1]],('Cayey','Caguas'):  [24,trafficType[0]],
('Caguas','Juncos'):  [15,trafficType[1]],
('Canovanas','Juncos'):  [21,trafficType[1]],('Canovanas','Fajardo'):  [31,trafficType[0]],
('Humacao','Fajardo'):  [36,trafficType[0]],('Humacao','Yabucoa'):  [18,trafficType[1]],
('Guayama','Salinas'):  [23,trafficType[0]],('Guayama','Yabucoa'):  [38,trafficType[1]]})


#Velocidades
# Ponce a Yauco = 65mph Ponce a Isablea = 55
# Mayaguez a Yauco = 65mph    Mayaguez a Rincon = 60  Mayaguez a Aguadilla = 65 Mayaguez a Maricao =35
# 


g = search.GraphProblem('Mayaguez', "SanJuan", PR_map)
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
