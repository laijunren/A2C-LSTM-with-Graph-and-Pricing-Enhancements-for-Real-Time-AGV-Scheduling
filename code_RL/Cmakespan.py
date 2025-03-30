# 导入 OR-Tools 中用于约束求解和车辆路径问题（VRP）的模块
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
# 导入 pandas 用于数据读取和处理
import pandas as pd
# 导入 numpy 用于数值计算（如构造距离矩阵）
import numpy as np
# 导入 math 模块，用于数学计算（例如开方）
import math

# 定义一个函数，用于计算两个坐标之间的欧几里得距离
def compute_euclidean_distance(coord1, coord2):
    """计算欧几里得距离"""
    # 使用生成器表达式计算每个坐标维度差的平方和，再开方返回欧几里得距离
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))

# 定义函数，用于创建数据模型，包含从 Excel 中读取任务数据以及构造节点和距离矩阵
def create_data_model():
    """创建数据模型，从 Excel 读取任务数据"""
    data = {}  # 用于存储所有数据

    # 设定车辆起始位置（Depot）的索引为 0，这里 depot 使用 'N021S00430' 对应的坐标
    data['depot'] = 0  

    # 定义各个节点的坐标，包含 depot 和其他任务相关的节点
    data['node_coords'] = {
        'N021S00430': (830, 2100, 40),    # Depot 的坐标
        'N021S00436': (1255, 1050, 40),
        'N021S00441': (1690, 1080, 40),
        'N021S00695': (1060, 1145, 40),
        'N0251S00501': (715, 1120, -40),
        'N0401S00230': (1450, 1020, 0),
        'N0401S00280': (1305, 1200, 0),
        'N074S01111': (1100, 1600, 40)
    }

    # 从 Excel 文件中加载任务数据
    file_path = "/home/aaa/my_code/hospital-main/Simulator/Hospital_DRL/test_instances/d5_164.xlsx" # 修改为你的文件路径
    df = pd.read_excel(file_path)  # 使用 pandas 读取 Excel 文件
    # 从 DataFrame 中提取任务数据，每个任务由一个 pickup 节点和一个 delivery 节点组成
    tasks = list(df[['start_node', 'end_node']].itertuples(index=False, name=None))
    data['tasks'] = tasks  # 将任务列表保存到数据模型中

    # 设置任务服务时间，单位为秒（这里每个任务的服务时间为 30 秒）
    data['service_time'] = 30

    # 固定车辆数量，定义使用 9 辆车辆（AGV）
    data['num_vehicles'] = 9

    # 构造完整的节点列表：第一个节点（索引 0）为 depot，后续依次为每个任务的 pickup 和 delivery 节点
    # Depot 使用 'N021S00430' 对应的坐标
    coords = [data['node_coords']['N021S00430']]
    # 遍历所有任务，依次添加 pickup 节点和 delivery 节点的坐标
    for (pickup, delivery) in tasks:
        # 检查任务中指定的节点是否在已定义的坐标字典中
        if pickup not in data['node_coords'] or delivery not in data['node_coords']:
            raise ValueError(f"任务节点 {pickup} 或 {delivery} 不在已定义的节点中")
        coords.append(data['node_coords'][pickup])
        coords.append(data['node_coords'][delivery])
    data['coords'] = coords  # 将节点坐标列表存储到数据模型中

    # 利用 NumPy 构造距离矩阵，矩阵的每个元素表示两个节点之间的欧几里得距离
    n = len(coords)  # 节点总数
    distance_matrix = np.zeros((n, n), dtype=int)  # 初始化 n×n 的矩阵，数据类型为整型
    for i in range(n):
        for j in range(n):
            # 计算节点 i 和节点 j 之间的距离，并将结果转换为整数存入矩阵中
            distance_matrix[i, j] = int(compute_euclidean_distance(coords[i], coords[j]))
    data['distance_matrix'] = distance_matrix.tolist()  # 将矩阵转换为列表形式存储

    return data  # 返回构造好的数据模型

# 定义一个函数用于打印求解结果，包括每辆车的路线和完成时间
def print_solution(data, manager, routing, solution, time_dimension):
    """输出每辆车的路线及完成时间"""
    total_time = 0  # 用于记录所有车辆中最长的路线时间（即 makespan）
    # 对于每一辆车进行遍历
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)  # 获取当前车辆的起始索引（对应 depot）
        route = []  # 用于存储当前车辆的路线节点
        # 遍历该车辆的路线，直到达到终点
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)  # 将内部索引转换为节点编号
            route.append(node)  # 添加当前节点到路线列表
            # 移动到下一个节点
            index = solution.Value(routing.NextVar(index))
        # 将终点（depot）也加入路线
        route.append(manager.IndexToNode(index))
        # 获取当前车辆完成最后一个节点时累积的时间，即该车辆的总路线时间
        route_time = solution.Value(time_dimension.CumulVar(index))
        # 输出当前车辆的路线和总耗时
        print(f"Route for vehicle {vehicle_id}: {route} | Route time: {route_time} seconds")
        # 更新 makespan，即所有车辆中最大的完成时间
        total_time = max(total_time, route_time)
    print(f"Makespan (max route time): {total_time} seconds")  # 输出 makespan

# 定义主函数，构造模型并求解
def main():
    data = create_data_model()  # 创建数据模型，包含任务、坐标、距离矩阵等信息
    num_tasks = len(data['tasks'])  # 任务数（每个任务包含 pickup 和 delivery）
    num_nodes = 1 + 2 * num_tasks  # 总节点数 = Depot + 2 * 任务数

    # 创建 RoutingIndexManager，用于管理节点索引、车辆数量以及 depot 节点
    manager = pywrapcp.RoutingIndexManager(num_nodes, data['num_vehicles'], data['depot'])
    # 创建 RoutingModel，基于上面定义的管理器构建车辆路径问题模型
    routing = pywrapcp.RoutingModel(manager)

    # 定义距离回调函数，用于返回两个节点之间的距离
    def distance_callback(from_index, to_index):
        # 将内部索引转换为实际节点编号，并返回对应的距离
        return data['distance_matrix'][manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
    # 注册回调函数，并获得回调函数的索引
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    # 设置所有车辆的弧成本评估器，即路径费用依据上面定义的距离回调函数计算
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 添加时间维度，使得求解器能够处理时间约束
    horizon = 30000  # 定义时间上限（最大允许时间）
    routing.AddDimension(
        transit_callback_index,  # 使用前面定义的距离回调函数作为时间消耗（可以理解为行驶时间）
        30,      # slack 时间（允许在节点处等待的时间，单位为秒）
        horizon, # 最大累积时间（时间窗上限）
        True,    # 强制所有车辆的累积时间从 0 开始（start cumul at zero）
        'Time'   # 维度名称
    )
    # 获取时间维度对象，用于后续时间约束的设置
    time_dimension = routing.GetDimensionOrDie('Time')
    # 设置全局跨度成本系数，目的是在目标中惩罚 makespan，鼓励缩短最大完成时间
    time_dimension.SetGlobalSpanCostCoefficient(100)

    # 添加 Pickup & Delivery 约束，确保每个任务的 pickup 和 delivery 成对出现
    for i in range(num_tasks):
        # 计算任务 i 对应的 pickup 节点在内部索引中的位置
        pickup_index = manager.NodeToIndex(1 + 2 * i)
        # 计算对应的 delivery 节点的内部索引位置
        delivery_index = manager.NodeToIndex(1 + 2 * i + 1)
        # 添加 Pickup & Delivery 约束，要求 pickup 和 delivery 必须出现在同一车辆的路径中
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        # 添加额外约束，确保 pickup 和 delivery 分配给同一辆车
        routing.solver().Add(routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index))
        # 添加时间约束，确保 delivery 节点的累计时间至少比 pickup 节点多 service_time（服务时间）
        routing.solver().Add(
            time_dimension.CumulVar(delivery_index) >= 
            time_dimension.CumulVar(pickup_index) + data['service_time']
        )

    # 设置求解参数
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.seconds = 600  # 设置求解时间上限为 600 秒（10 分钟）
    # 使用局部最优插入策略作为初始解策略
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION
    # 使用 GUIDED_LOCAL_SEARCH 作为局部搜索元启发式方法以改善初始解
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

    # 开始求解
    solution = routing.SolveWithParameters(search_parameters)
    # 如果找到解，则打印解的详细信息，否则输出未找到解的信息
    if solution:
        print_solution(data, manager, routing, solution, time_dimension)
    else:
        print("No solution found!")

if __name__ == '__main__':
    main()
