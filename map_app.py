import mainCode
import search

cities = "SanJuan  -  Ponce  -  Salinas  -  Adjuntas   -  Yauco  -  Maricao   -  Rincon   -  Auadilla   -  Camuy   -  Utuado   -  Arecibo   -  Bayamon   -  Catano   -  Cayey   -  Caguas   -  Juncos   -  Canovanas   -  Humacao   -  Yabucoa   -  Guayama   -  Fajardo,   -  Mayaguez"  
def run_app():
    print("Options to travel from and towards to: ",  cities)
    initial = input("Enter your current town: ")
    if initial not in mainCode.locs:
        print("City not in database, please try again.")
        return run_app()
    goal = input("Enter your goal destination: ")
    if goal not in mainCode.locs:
        print("City not in database, please try again.")
        return run_app()
    
    BUCS = search.bidirectional_uniform_cost_search(search.RouteProblem(initial, goal, map=mainCode.PR_map))
    print("The estimated time is ", BUCS.path_cost,"minutes. The optimal path is:",search.path_states(BUCS),'.' )
    return
print(run_app())
