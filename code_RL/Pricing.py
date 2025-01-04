import gurobipy as gp


def powerset(s):
    """Generate all possible subsets of a set"""
    result = [[]]
    for x in s:
        # For each existing subset, create a new subset by adding x
        new_subsets = [subset + [x] for subset in result]
        result.extend(new_subsets)
    return result


def route_pairs(route):
    """Generate pairs of arcs from a route"""
    pairs = []
    # Iterate over the first and second-to-last arcs of the route
    for i in range(len(route) - 1):
        # Add the current arc and the next arc to the pairs list
        pairs.append((route[i], route[i + 1]))
    # Add the last arc of the route to the pairs list
    pairs.append((route[-1], route[0]))  # Change this line to avoid the depot at the end
    return pairs


def euclidian_distance(x1, y1, z1, x2, y2, z2):
    return (((x1 - x2) / 1000) ** 2 + ((y1 - y2) / 1000) ** 2 + ((z1 - z2) / 10) ** 2) ** 0.5


def pricing(map):
    # Define the problem parameters
    depot = 'Depot'
    nodes = ['N0', 'L1', 'L2', 'L3']
    num_vehicle = 9
    ### 1997 950
    ### 1112 946
    ### 1515 947

    map += [[1997, 950, 0], [1112, 946, 0], [1515, 947, 0]]
    # Assuming a symmetric distance matrix for simplicity
    distance_matrix = {}
    for node1 in nodes + [depot]:
        for node2 in nodes + [depot]:
            if node1 == 'Depot' and node2 != 'Depot':
                distance_matrix[(node1, node2)] = euclidian_distance(
                    0, 0, 0,
                    map[nodes.index(node2)][0], map[nodes.index(node2)][1], map[nodes.index(node2)][2])
            elif node1 != 'Depot' and node2 == 'Depot':
                distance_matrix[(node1, node2)] = euclidian_distance(
                    map[nodes.index(node1)][0], map[nodes.index(node1)][1], map[nodes.index(node1)][2],
                    0, 0, 0)
            elif node1 != node2:  # Skip the same node (a request of pickup and delivery, or lift transfer) pair
                distance_matrix[(node1, node2)] = euclidian_distance(
                    map[nodes.index(node1)][0], map[nodes.index(node1)][1], map[nodes.index(node1)][2],
                    map[nodes.index(node2)][0], map[nodes.index(node2)][1], map[nodes.index(node2)][2])

    # Generate all possible routes for each vehicle
    possible_routes = []
    for salesman in range(num_vehicle):
        for subset in powerset(nodes):
            if subset:  # Exclude the empty set
                route = tuple(subset) + (depot,)
                # Check if the route is already in the list
                if route not in possible_routes:
                    possible_routes.append(route)

    # Create a Gurobi model
    model = gp.Model('VRPPD_Hospital_SetCovering')
    model.Params.OutputFlag = 0

    # Create variables
    x = model.addVars(possible_routes, vtype=gp.GRB.CONTINUOUS)
    num_visits = {node: model.addVar(vtype=gp.GRB.CONTINUOUS, name=f'num_visits_{node}') for node in nodes}

    # Set the objective function
    obj = gp.quicksum(distance_matrix[arc] * x[route] for route in possible_routes for arc in route_pairs(route))
    model.setObjective(obj, gp.GRB.MINIMIZE)

    # Add constraints
    # # Each node must be visited exactly once
    # for node in nodes:
    #     model.addConstr(gp.quicksum(x[route] for route in possible_routes if node in route) == 1)

    # Each node must be visited at least once -- visit
    '''The dual value for the constraint that each node must be visited at least once represents the marginal cost of visiting 
    that node once more than the minimum required. If the dual value is positive, it suggests that visiting the node more 
    than once provides a benefit to the objective function (e.g., additional revenue from moving more vehicles by that lift, 
    therefore, from the constraint, we only cares the dual values of the lift, for example, a route, N1 -1 N2 -2 N3 -1 N4 -1...,
    we only interested in the dual value of -1 or -2, representing dual values of using lift 1 or lift 2). 
    If the dual value is negative, it suggests that visiting the node more than once would be detrimental to the objective 
    function (e.g., additional costs associated with visiting the node more than once).'''
    for node in nodes:
        model.addConstr(num_visits[node] >= 1, f'{node}_visit')

    # Each node must be covered in the solution -- demand
    '''The dual value for the constraint that each node must be covered in the solution represents the marginal benefit of covering 
    that node in the solution. If the dual value is positive, it suggests that including that node in the solution provides 
    a benefit to the objective function (e.g., additional revenue from serving that request). If the dual value is negative, 
    it suggests that including that node/request in the solution would be detrimental to the objective function (e.g., additional costs 
    associated with serving that node).'''
    for node in nodes:
        model.addConstr(gp.quicksum(x[route] for route in possible_routes if node in route) >= 1, f'{node}_demand')

    # Solve the model
    model.optimize()

    # Retrieve the dual values for the constraints
    dual_values = {}
    for constr in model.getConstrs():
        dual_values[constr.constrName] = constr.getAttr('Pi')

    node_duals = {}
    # Iterate over the constraints and assign the dual value to the corresponding constraints
    for constraint in dual_values:
        node_duals[constraint] = dual_values[constraint]

    # for node, dual in node_duals.items():
    #     print(f"Dual value for constraint on node {node}: {dual}")

    # return the values of demands in node_duals for the constraints
    return node_duals['N0_demand'], node_duals['L1_demand'], node_duals['L2_demand'], node_duals['L3_demand']


if __name__ == '__main__':
    # Define the map
    map = [[1000, 1000, 10]]

    # Call the pricing function
    pricing(map)


