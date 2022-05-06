import mainCode
import search

def run_app():
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
