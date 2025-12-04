import math
import cvxpy as cp
import numpy as np
import random

def init_param_hetero(constant, n, t):
    parameters = {'sigma' : constant['sigma'], 'D_n': np.empty(n), 'Gamma': constant['Gamma'], 'local_iter': np.empty(n), 'c_n': np.empty(n),
                  'frequency_n' : np.empty(n), 'weight_size_n' : np.empty(n),
                  'number_of_clients' : n, 'bandwidth' : np.empty(n), 'channel_gain_n': np.empty(n), 'transmission_power_n' : np.empty(n), 'noise_W' : np.empty(n),
                  'transmission_rate' : np.empty(n), 't': t}

    # Scaling factors for unit conversion
    scale_bandwidth = 1e6  # Convert MHz to Hz
    scale_MB_to_bit = 8 * 1e6 # Convert MB to bit
    
    for i in range(constant['number_of_clients']):
        # 이질성 설정 수정
        if len(constant["frequency_n_GHz"]) > 1:
            f_i = i % len(constant["frequency_n_GHz"])
        else:
            f_i = 0
            
        if len(constant["transmission_power_n"]) > 1:
            t_i = i % len(constant["transmission_power_n"])
        else:
            t_i = 0
            
        parameters["sigma"] = constant["sigma"]
        parameters["D_n"][i] = constant["D_n"][i]
        parameters["local_iter"][i] = constant["local_iter"]
        parameters["c_n"][i] = constant["c_n"]
        parameters["frequency_n"][i] = constant["frequency_n_GHz"][f_i] * 1e9  # GHz를 Hz로 변환
        parameters["weight_size_n"][i] = constant["weight_size_n_kbit"] * 1e3  # Kbits를 bits로 변환

        # Calculate R
        parameters["number_of_clients"] = constant["number_of_clients"]
        parameters["bandwidth"][i] = constant["bandwidth_MHz"] * 1e6  # MHz를 Hz로 변환
        parameters["channel_gain_n"][i] = constant["channel_gain_n"]
        parameters["transmission_power_n"][i] = constant["transmission_power_n"][t_i]
        parameters["noise_W"][i] = constant["noise_W"]
        # Shannon capacity formula: R = (B/N) * log2(1 + P*h/N0)
        parameters["transmission_rate"][i] = parameters["bandwidth"][i]/parameters["number_of_clients"] \
            * math.log2(1 + parameters["channel_gain_n"][i] * parameters["transmission_power_n"][i] / parameters["noise_W"][i])

    return parameters


def objective_function(v_n, t, r, parameters):
    # Objective: maximize sum(v_n * D_n) / t (minimize negative)
    # Discounting factor sigma^r is applied for multi-round optimization
    if r == 0:
        # First round: no discounting
        objective = -cp.sum(v_n @ parameters['D_n']) / t
    else:
        objective = -parameters['sigma']**r * cp.sum(v_n @ parameters['D_n']) / t
    return objective

def result_function(v_n, t, r, parameters):
    objective = -parameters['sigma']**(r-1) * \
        sum(v_n[i] * parameters['D_n'][i] for i in range(parameters["number_of_clients"])) / t
    return objective

def block_coordinate_descent(parameters, round, t):
    n = parameters['number_of_clients']

    # 디버그 출력
    print(f"\n=== ADM Debug Round {round} ===")
    for i in range(n):
        print(f"Client {i}: ")
        print(f"  frequency: {parameters['frequency_n'][i]/1e9:.2f} GHz")
        print(f"  c_n: {parameters['c_n'][i]}")
        print(f"  D_n: {parameters['D_n'][i]}")
        print(f"  transmission_rate: {parameters['transmission_rate'][i]/1e6:.2f} Mbps")
        print(f"  weight_size: {parameters['weight_size_n'][i]/1e3:.1f} Kbits")
        print(f"  Gamma: {parameters['Gamma']}")
        print(f"  t: {t} sec")
        # Constraint 확인
        comp_time = parameters['local_iter'][i] * parameters['c_n'][i] * 1.0 * parameters['D_n'][i] / parameters['frequency_n'][i]
        comm_time = parameters['weight_size_n'][i] / parameters['transmission_rate'][i]
        print(f"  Constraint check (v_n=1.0): comp={comp_time:.4f}s + comm={comm_time:.4f}s = {comp_time+comm_time:.4f}s (must <= {t}s)")
        

    # Define variables
    v_n = cp.Variable(n)

    # Define constraints for each block
    constraints = []
    for i in range(n):
        block_constraints = [
            parameters['Gamma'] <= v_n[i],
            v_n[i] <= 1,
            parameters['local_iter'][i] * parameters['c_n'][i] * v_n[i] * parameters['D_n'][i] /\
                 parameters['frequency_n'][i] + parameters['weight_size_n'][i] /\
                     parameters['transmission_rate'][i] <= t
        ]
        constraints += block_constraints
    
    t_optimal = t
    # Block coordinate descent 반복
    max_iter = 50
    tolerance = 1e-12
    pre_sol = 9999999
    sol_list = []

    # Solve the optimization problem using block coordinate descent
    for iteration in range(max_iter):  # Set the desired number of iterations
        v_n_objective = cp.Minimize(objective_function(v_n, t_optimal, round, parameters))
        v_n_problem = cp.Problem(v_n_objective, constraints)

        # Try multiple solvers
        solved = False
        for solver in [cp.ECOS, cp.SCS, cp.CLARABEL]:
            try:
                v_n_problem.solve(solver=solver, verbose=False)
                if v_n.value is not None and v_n_problem.status in ['optimal', 'optimal_inaccurate']:
                    solved = True
                    break
            except:
                continue

        if not solved or v_n.value is None:
            print(f"[WARNING] Solver failed at round {round}, iteration {iteration}. Using default v_n = [1.0, 1.0]")
            v_n_optimal = [1.0 for _ in range(n)]
            t_optimal = t
            sol = 0
            sol_list.append(sol)
            break

        v_n_optimal = v_n.value
        max_t = []
        for i in range(n):
            tmp = [parameters['local_iter'][i] * parameters['c_n'][i] * v_n_optimal[i] *\
                    parameters['D_n'][i] / parameters['frequency_n'][i] + \
                        parameters['weight_size_n'][i] / parameters['transmission_rate'][i]]
            max_t += tmp

        t_optimal = max(max_t)
        sol = result_function(v_n_optimal, t_optimal, round, parameters)
        sol_list.append(sol)

    v_n_optimal = [float(x) for x in v_n_optimal]

    return v_n_optimal, sol_list, t_optimal

def descent_01(parameters, round, t):
    n = parameters['number_of_clients']
    v_n = [1.0 for _ in range(n)]
    
    t_optimal = t
    # Block coordinate descent 반복
    max_iter = 1
    sol_list = []

    # Solve the optimization problem using block coordinate descent
    for r in tqdm(range(round)):
        for _ in range(max_iter):  # Set the desired number of iterations
            max_t = []
            for i in range(n):
                tmp = [parameters['local_iter'][i] * parameters['c_n'][i] * v_n[i] *\
                        parameters['D_n'][i] / parameters['frequency_n'][i] + \
                            parameters['weight_size_n'][i] / parameters['transmission_rate'][i]]
                max_t += tmp

            t_optimal = max(max_t)
            sol = result_function(v_n, t_optimal, r, parameters)

        sol_list.append(sol)
        if r % 15 == 0:
            v_n = [x -0.1 for x in v_n]

    return sol_list

if __name__ == "__main__":
    # Example usage - 논문 파라미터 기반
    constant_parameters = {
        'sigma': 0.9 * 1e-8,  # Discounting factor
        'D_n': [2500] * 20,   # Data samples per client (can vary)
        'Gamma': 0.4,         # Gamma parameter from paper
        'local_iter': 10,     # Local iterations (E)
        'c_n': 30,            # CPU cycles per sample
        'frequency_n_GHz': [3],  # 3 GHz computation capacity
        'weight_size_n_kbit': 100,  # 100 Kbits model size
        'number_of_clients': 20,  # N = 20 MDs
        'bandwidth_MHz': 1,      # Bandwidth
        'channel_gain_n': 1,     # Channel gain
        'transmission_power_n': [1],  # Transmission power
        'noise_W': 10**(-114/10) * 1e-3  # -114 dBm noise power
    }

    parameters = init_param_hetero(constant_parameters, constant_parameters['number_of_clients'], 500)
    
    r = 1
    t = parameters['t']
    optimal_v_n, sol_list, optimal_t = block_coordinate_descent(parameters, r, t)
    print(optimal_v_n)
    print(optimal_t)
