from time import time
import search
from datetime import datetime

t = datetime.now()
time = int(t.strftime("%H"))
trafficType = ['heavy','mixed','light'] 

def arrivalTime(dist, vel, trafficVol):
    kph =  vel*1.609
    opmltrvltime = dist/kph
    if (time >=7 and time<=9) or (time >=15 and time<=18):
        if trafficVol == 'heavy':
            return opmltrvltime * 1.5
        elif trafficVol == 'mixed':
            return opmltrvltime * 1.26
    else:
        return  opmltrvltime

PR_map = search.Map({('Ponce','Salinas'):  [36,50,trafficType[0]],('Ponce','Adjuntas'):  [25,40,trafficType[2]],('Ponce','Yauco'):  [32,55,trafficType[0]],
('Mayaguez','Yauco'):  [45,55,trafficType[0]],('Mayaguez','Maricao'):  [26,40,trafficType[1]],('Mayaguez','Rincon'):  [23,50,trafficType[0]],('Mayaguez','Aguadilla'):  [28,55,trafficType[0]],
('Rincon','Aguadilla'):  [17,55,trafficType[0]],('Camuy','Aguadilla'):  [39,50,trafficType[0]],('Camuy','Utuado'):  [44,40,trafficType[2]],('Camuy','Arecibo'):  [15,50,trafficType[0]],
('Utuado','Arecibo'):  [33,50,trafficType[1]],('Utuado','Adjuntas'):  [21,35,trafficType[2]],('Adjuntas','Maricao'):  [54,35,trafficType[2]],
('Bayamon','Arecibo'):  [77,50,trafficType[0]],('Bayamon','Catano'):  [11,45,trafficType[1]],('Bayamon','SanJuan'):  [19,55,trafficType[0]],
('SanJuan','Cayey'):  [55,55,trafficType[0]],('SanJuan','Caguas'):  [33,55,trafficType[0]],('SanJuan','Canovanas'):  [30,50,trafficType[1]],
('Cayey','Salinas'):  [32,50,trafficType[1]],('Cayey','Guayama'):  [42,55,trafficType[1]],('Cayey','Caguas'):  [24,55,trafficType[0]],
('Caguas','Juncos'):  [15,50,trafficType[1]],
('Canovanas','Juncos'):  [21,50,trafficType[1]],('Canovanas','Fajardo'):  [31,50,trafficType[0]],
('Humacao','Fajardo'):  [36,55,trafficType[0]],('Humacao','Yabucoa'):  [18,55,trafficType[1]],
('Guayama','Salinas'):  [23,50,trafficType[0]],('Guayama','Yabucoa'):  [38,50,trafficType[1]]})



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
