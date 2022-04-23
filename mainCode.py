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



locs = {'SanJuan':(2044227.08, 805685.13), 'Ponce':(1993038.45,752625.54),'Salinas':(1989776.68,786168.85),
        'Adjuntas':(2009766.0,740860.58),'Yauco':(1995376.04,727613.80), 'Maricao':(2010542.0,717323.0),
        'Rincon':(2028715.19, 684922.70), 'Aguadilla':(2038469.91,694965.94),'Camuy':(2041508.70,726682.74),
        'Utuado':(2021090.65,743122.28), 'Arecibo':(2040952.74, 748582.15),'Bayamon':(2035606.90, 799519.64),
        'Catano':(2041984.68, 802564.11), 'Cayey':(2004874.15,799924.19), 'Caguas':(2019146.30,813544.32),
        'Juncos':(2017815.84, 191065.79),'Canovanas':(2034106.34,193574.50),
        'Humacao':(2009046.63, 200840.64), 'Yabucoa':(1998149.64, 195169.11), 'Guayama':(1990808.29, 805675.48),
        'Fajardo':(2028271.61, 219651.35), 'Mayaguez':(2013453.29,696161.94)}
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
('Guayama','Salinas'):  [23,50,trafficType[0]],('Guayama','Yabucoa'):  [38,50,trafficType[1]]},locs)

#locs= 'City':(UTM_Northing, UTM_Easting)


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
