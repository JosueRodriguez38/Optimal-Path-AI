import search

PR_map = search.Map({('Ponce','Salinas'):  36,('Ponce','Adjuntas'):  25,('Ponce','Yauco'):  32,
('Mayaguez','Yauco'):  45,('Mayaguez','Maricao'):  26,('Mayaguez','Rincon'):  23,('Mayaguez','Aguadilla'):  28,
('Rincon','Aguadilla'):  17,('Camuy','Aguadilla'):  39,('Camuy','Utuado'):  44,('Camuy','Arecibo'):  15,
('Utuado','Arecibo'):  33,('Utuado','Adjuntas'):  21,('Adjuntas','Maricao'):  54,
('Bayamon','Arecibo'):  77,('Bayamon','Catano'):  11,('Bayamon','SanJuan'):  19,
('SanJuan','Cayey'):  55,('SanJuan','Caguas'):  33,('SanJuan','Canovanas'):  30,
('Cayey','Salinas'):  32,('Cayey','Guayama'):  42,('Cayey','Caguas'):  24,
('Caguas','Juncos'):  15,
('Canovanas','Juncos'):  21,('Canovanas','Fajardo'):  31,
('Humacao','Fajardo'):  36,('Humacao','Yabucoa'):  18,
('Guayama','Salinas'):  23,('Guayama','Yabucoa'):  38,})


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
