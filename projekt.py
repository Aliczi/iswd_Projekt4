import numpy as np
from pulp import *

def load_data(filename):
    data = []
    with open(filename) as f:
        column_names = f.readline().replace("\n","").split(';')

        for line in f.readlines():
            data.append(line.replace("\n","").split(";"))
    # return column_names, data
    return np.array(column_names), np.array(data)

def load(filename):
    col_names, data = load_data(filename)
    col_names = col_names[1:]
    city_names = data[:,0]
    data = data[:,1:]

    return data.astype(float), city_names, col_names

def load_samples(filename):
    col_names, data = load_data(filename)

    col_names = col_names[1:]
    data = data[:, 1:]
    
    for i,name in enumerate(col_names):
        if name[0] == 'o':
            border = i
            break
    
    return data[:,:border].astype(float), data[:,border:].astype(float)



# input-oriented combination-based CCR Model
def calc_eff(inputs, outputs, index, names = None, do_print = False, do_super_eff = False):
    model = LpProblem(name='eff', sense=LpMinimize)
    teta = LpVariable(name='teta' , lowBound=0, cat='Continuous')
    model += teta

    units_size = inputs.shape[0]
    inputs_size = inputs.shape[1]
    outputs_size = outputs.shape[1]

    if names is None:
        lambdas = [LpVariable(name=f'lambda_{i}' , lowBound=0, cat='Continuous')for i in range(units_size)]
    else:
        lambdas = [LpVariable(name=f'lambda_{names[i]}' , lowBound=0, cat='Continuous')for i in range(units_size)]
    
    # inputs
    for input_idx in range(inputs_size):
        in_sum = []
        for unit_idx in range(units_size):
            if do_super_eff and unit_idx == index:
                continue
            in_sum.append(lambdas[unit_idx] * inputs[unit_idx][input_idx])
        in_sum = sum(in_sum)
        model += in_sum <= inputs[index][input_idx] * teta
        # print(in_sum <= inputs[index][input_idx] * teta)
    
    # outputs
    for output_idx in range(outputs_size):
        out_sum = []
        for unit_idx in range(units_size):
            if do_super_eff and unit_idx == index:
                continue
            out_sum.append(lambdas[unit_idx] * outputs[unit_idx][output_idx])
        out_sum = sum(out_sum)
        model += out_sum >= outputs[index][output_idx]
        # print(out_sum >= outputs[index][output_idx])
    # limits
    for l in lambdas:
        model += l >= 0
    model += teta >= 0

    model.solve(PULP_CBC_CMD(msg=0))

    if do_print:
        print(f'status: {model.status}, {LpStatus[model.status]}')
        print(f"teta = {teta.value()}")
        for l in lambdas:
            print(f"{l.name} = {l.value()}")
    
    lambdas_values = [l.value() for l in lambdas]

    return teta.value(), lambdas_values

# input-oriented efficiency-based CCR Model
def calc_eff2(inputs, outputs, index, do_print = False, do_super_eff = False):
    model = LpProblem(name='eff', sense=LpMaximize)
    
    units_size = inputs.shape[0]
    inputs_size = inputs.shape[1]
    outputs_size = outputs.shape[1]

    u_list = [LpVariable(name=f'u_{i}' , lowBound=0, cat='Continuous')for i in range(outputs_size)]
    v_list = [LpVariable(name=f'v_{i}' , lowBound=0, cat='Continuous')for i in range(inputs_size)]

    for unit_idx in range(units_size):
        in_sum = []
        for input_idx in range(inputs_size):
            in_sum.append(v_list[input_idx]*float(inputs[unit_idx][input_idx]))
        
        out_sum = []
        for output_idx in range(outputs_size):
            out_sum.append(u_list[output_idx]*float(outputs[unit_idx][output_idx]))
        
        in_sum = sum(in_sum)
        out_sum = sum(out_sum)
        if not (do_super_eff and unit_idx == index): 
            model += out_sum <= in_sum
        # print(out_sum <= in_sum)

        if unit_idx == index:
            target_in_sum = in_sum
            target_out_sum = out_sum
            model += out_sum
            model += in_sum == 1

    for u in u_list:
        model += u >= 0
    
    for v in v_list:
        model += v >= 0
    
    model.solve(PULP_CBC_CMD(msg=0))

    eff = target_out_sum.value()
    
    if do_print:
        print(f"eff = {eff}")
        for u in u_list:
            print(f"{u.name} = {u.value()}")
        for v in v_list:
            print(f"{v.name} = {v.value()}")

    u_values = [u.value() for u in u_list]
    v_values = [v.value() for v in v_list]

    return eff, u_values, v_values

def calc_weights_for_cross_efficiency(inputs, outputs, index, efficiency, agressive = True):
    if agressive:
        model = LpProblem(name='eff', sense=LpMinimize)
    else:
        model = LpProblem(name='eff', sense=LpMaximize)


    units_size = inputs.shape[0]
    inputs_size = inputs.shape[1]
    outputs_size = outputs.shape[1]

    u_list = [LpVariable(name=f'u_{i}' , lowBound=0, cat='Continuous')for i in range(outputs_size)]
    v_list = [LpVariable(name=f'v_{i}' , lowBound=0, cat='Continuous')for i in range(inputs_size)]

    
    for unit_idx in range(units_size):
        
        in_sum = []
        for input_idx in range(inputs_size):
            in_sum.append(v_list[input_idx]*float(inputs[unit_idx][input_idx]))
        
        out_sum = []
        for output_idx in range(outputs_size):
            out_sum.append(u_list[output_idx]*float(outputs[unit_idx][output_idx]))
        
        in_sum = sum(in_sum)
        out_sum = sum(out_sum)
        
        if unit_idx == index:
            model += out_sum == efficiency*(in_sum)
            # print(out_sum == efficiency*(in_sum))
        else:    
            model += out_sum <= in_sum
            # print(out_sum <= in_sum)
    
    # combine unit
    out_sum = []
    for output_idx in range(outputs_size):
        combined_value = np.sum(outputs[:,output_idx]) - outputs[index][output_idx]
        out_sum.append(u_list[output_idx]*combined_value)
    model += sum(out_sum) 
    # print(sum(out_sum))

    in_sum = []
    for input_idx in range(inputs_size):
        combined_value = np.sum(inputs[:,input_idx]) - inputs[index][input_idx]
        in_sum.append(v_list[input_idx]*combined_value)
    model += sum(in_sum) == 1
    # print(sum(in_sum) == 1)

    for u in u_list:
        model += u >= 0
    
    for v in v_list:
        model += v >= 0

    model.solve(PULP_CBC_CMD(msg=0))

    u_values = [u.value() for u in u_list]
    v_values = [v.value() for v in v_list]

    return v_values, u_values

# row of cross-efficiency matrix
def calc_efficiency_from_weights(inputs, outputs, weights_input, weights_output):
    eff_list = []
    for out in outputs:
        eff = 0
        for o, w in zip(out,weights_output):
            eff+= o*w

        eff_list.append(eff)
    
    for i, inp in enumerate(inputs):
        eff = 0
        for o, w in zip(inp,weights_input):
            eff+= o*w

        eff_list[i]/=eff

    return eff_list

def calc_cross_efficiency_matrix(inputs, outputs):
    matrix = []

    for unit_idx in range(len(outputs)):
        eff, _ = calc_eff(inputs, outputs, unit_idx)

        v_w, u_w = calc_weights_for_cross_efficiency(inputs,outputs,unit_idx,eff)

        matrix.append(calc_efficiency_from_weights(inputs,outputs, v_w, u_w))
    return matrix

def calc_cross_efficiency(inputs, outputs, roundDecimals = None):
    cross_matrix = np.array(calc_cross_efficiency_matrix(inputs, outputs))
    # cross_matrix = np.round(cross_matrix,3)
    # print(cross_matrix)
    cross_efficiency = np.mean(cross_matrix,axis=0) 
    # print(cross_entropy)
    if not roundDecimals is None:
        cross_matrix = np.round(cross_matrix,roundDecimals)
        cross_efficiency = np.round(cross_efficiency,roundDecimals)
    return cross_efficiency, cross_matrix
    
def calc_monte_carlo(inputs, outputs, inputs_weights, outputs_weights, interval_num = 5):
    eff_list = [calc_efficiency_from_weights(inputs, outputs, in_w, out_w) for in_w, out_w in zip(inputs_weights, outputs_weights)]
    eff_list = np.array(eff_list)
    
    eff_mean = []

    inter_all = []
    for unit_idx in range(len(inputs)):
        intervals = [0]*interval_num
        eff = eff_list[:,unit_idx]

        eff_mean.append(np.mean(eff))
        
        for i in range(interval_num):
            intervals[i] = np.sum((eff>= i/interval_num) & (eff < (i+1)/interval_num))
        intervals[-1] += np.sum(eff >= 1)
        inter_all.append(intervals)
        
    return np.array(inter_all)/len(inputs_weights[:,0]), eff_mean 

if __name__ == "__main__":
    inputs, units_names, _ = load("inputs.csv")
    outputs, _, _ = load("outputs.csv")
    inputs, units_names, _ = load("inputs_test.csv")
    outputs, _, _ = load("outputs_test.csv")
    # print(calc_eff2(inputs, outputs, ids, do_super_eff= True))
    # print(calc_eff(inputs, outputs, ids, do_super_eff= True))
    
    for unit_idx in range(len(units_names)):
        eff, lambdas_values= calc_eff(inputs, outputs, unit_idx, do_super_eff=False)
        
        text = ""

        if eff == 1:
            super_eff, _ = calc_eff(inputs, outputs, unit_idx, do_super_eff=True)
            text += f"super eff = {super_eff} "
        
        for i, lv in enumerate(lambdas_values):
            if lv > 0 and i != unit_idx:
                text += f"({lv}){units_names[i]} "
        print(f"{units_names[unit_idx]} eff = {eff} {text}")

    print("\ncross_efficiency:")
    cross_eff, cross_eff_matrix = calc_cross_efficiency(inputs,outputs, 2)
    print(cross_eff)
    print("\nmatrix:")
    print(cross_eff_matrix)
    
    sample_inputs, sample_outputs = load_samples("samples_homework.csv")
    # print(sample_outputs)
    print("\nsamplowanie")
    matrix, EE = calc_monte_carlo(inputs, outputs, sample_inputs,sample_outputs)

    print(matrix)
    print("\n", EE)