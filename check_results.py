
import json

run_names = [
    'LR=1e-4_t=0', 'LR=1e-4_t=1', 'LR=1e-5_t=0', 'LR=1e-5_t=1'
]

with open(f'./model/{run_names[0]}/poorly_predicted.json', 'r') as file:
    run0_dict = json.load(file)
poorly_predicted_in_all = {}
for key in run0_dict.keys():
    poorly_predicted_in_all[key] = True

for run_name in run_names:
    with open(f'./model/{run_name}/poorly_predicted.json', 'r') as file:
        pp = json.load(file)

    for obj in pp.keys():
        if pp[obj] == False:
            poorly_predicted_in_all[key] = False
    
    with open(f'./model/{run_name}/material_log_acc.json', 'r') as file:
        mat = json.load(file)
    with open(f'./model/{run_name}/shape_log_acc.json', 'r') as file:
        shap = json.load(file)

    print(run_name)
    print(mat)
    print(shap)
    print('\n')

print('Poorly predicted in all:')
for obj in poorly_predicted_in_all:
    if poorly_predicted_in_all[obj]:
        print(obj)
print('\n')