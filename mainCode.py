import search
locs = {'SanJuan':(2044227.08, 805685.13), 'Ponce':(1993038.45,752625.54),'Salinas':(1989776.68,786168.85),
        'Adjuntas':(2009766.0,740860.58),'Yauco':(1995376.04,727613.80), 'Maricao':(2010542.0,717323.0),
        'Rincon':(2028715.19, 684922.70), 'Aguadilla':(2038469.91,694965.94),'Camuy':(2041508.70,726682.74),
        'Utuado':(2021090.65,743122.28), 'Arecibo':(2040952.74, 748582.15),'Bayamon':(2035606.90, 799519.64),
        'Catano':(2041984.68, 802564.11), 'Cayey':(2004874.15,799924.19), 'Caguas':(2019146.30,813544.32),
        'Juncos':(2017815.84, 191065.79),'Canovanas':(2034106.34,193574.50),
        'Humacao':(2009046.63, 200840.64), 'Yabucoa':(1998149.64, 195169.11), 'Guayama':(1990808.29, 805675.48),
        'Fajardo':(2028271.61, 219651.35), 'Mayaguez':(2013453.29,696161.94)}
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
('Guayama','Salinas'):  23,('Guayama','Yabucoa'):  38,},locs)
#locs= 'City':(UTM_Northing, UTM_Easting)



#Velocidades
# Ponce a Yauco = 65mph Ponce a Isablea = 55
# Mayaguez a Yauco = 65mph    Mayaguez a Rincon = 60  Mayaguez a Aguadilla = 65 Mayaguez a Maricao =35
#


g = search.RouteProblem('Mayaguez', "SanJuan", map =PR_map)
p1 = search.breadth_first_search(g)
print("This is breadth first search", p1)
p2 = search.uniform_cost_search(g)
print("This is uniform cost search", p2)
p3 = search.depth_first_recursive_search(g)
print("This is depth cost search", p3)
p4 = search.greedy_bfs(g)
#, lambda n: g.h(n)
print("This is greedy search", p4)
#lambda n: g.h(n) + n.path_cost
p5 = search.astar_search(g)
print("This is a* search", p5)

print(search.straight_line_distance(locs['Ponce'],locs['Mayaguez']))
print(search.straight_line_distance(locs['Adjuntas'],locs['Mayaguez']))
print(search.straight_line_distance(locs['SanJuan'],locs['Mayaguez']))
print(search.straight_line_distance(locs['Ponce'],locs['Salinas']))
