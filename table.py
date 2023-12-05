from collections import defaultdict
import numpy as np
raw = """  DiffHOI			iHOI			HHOR		
	F5	F10	CD	F5	F10	CD	F5	F10	CD
Mug	0.64	0.86	1.01	0.44	0.71	2.10	0.18	0.37	6.95
Bottle	0.54	0.92	0.72	0.48	0.77	1.48	0.26	0.56	3.09
Kettle	0.43	0.77	1.50	0.21	0.45	6.31	0.12	0.30	11.35
Bowl	0.79	0.98	0.36	0.38	0.64	3.10	0.31	0.54	4.18
Knife	0.50	0.95	0.76	0.33	0.68	2.82	0.71	0.93	0.63
ToyCar	0.83	0.99	0.27	0.66	0.95	0.54	0.26	0.59	1.90
mean	0.62	0.91	0.77	0.42	0.70	2.73	0.31	0.55	4.68
"""

metrics = ['F5','F10','CD', 'CD_h', 'MJMPE', "AUC"]
categories = ['Mug','Bottle','Kettle','Bowl','Knife','ToyCar']
data = {}
data['DiffHOI'] = """
0.606	0.876	0.866	26.619	1.141	0.772
0.635	0.97	0.533	14.68	0.976	0.805
0.606	0.883	1.193	41.557	1.106	0.784
0.733	0.954	0.483	73.041	1.195	0.761
0.783	0.983	0.381	82.86	1.354	0.729
0.791	0.984	0.319	30.492	1.219	0.756
0.682	0.839	1.147	9.652	0.976	0.805
0.443	0.876	0.91	15.941	1.041	0.792
0.249	0.652	1.805	42.769	1.03	0.794
0.844	1	0.245	130.481	1.217	0.757
0.221	0.925	1.13	100.298	1.313	0.738
0.871	0.996	0.228	16.092	0.862	0.828
"""

data['iHOI'] = """
0.456	0.718	2.038	18.317	1.20	0.76
0.507	0.804	1.335	14.636	1.13	0.77
0.18	0.39	7.517	39.263	1.22	0.76
0.292	0.539	4.509	35.408	1.37	0.73
0.357	0.763	1.61	17.913	1.36	0.73
0.664	0.95	0.53	26.919	1.37	0.73
0.423	0.7	2.167	21.028	1.01	0.80
0.443	0.734	1.617	13.21	1.06	0.79
0.247	0.5	5.104	32.472	1.00	0.80
0.474	0.744	1.69	63.234	1.08	0.79
0.305	0.593	4.035	25.965	1.42	0.72
0.659	0.943	0.549	16.277	1.03	0.79
"""

data['HHOR'] = """
0.134	0.287	7.672	353.906
0.322	0.671	2.413	25.324
0.11	0.277	12.708	44.038
0.326	0.585	4.007	53.456
0.767	0.973	0.374	221.7
0.233	0.556	2.073	318.24
0.223	0.444	6.225	104.354
0.191	0.441	3.761	318.698
0.133	0.319	9.988	156.747
0.294	0.5	4.357	46.782
0.658	0.893	0.88	148.545
0.291	0.632	1.721	193.371
"""

data['Conditional'] = """
0.6	0.863	0.963	12.587	1.178	0.764
0.577	0.967	0.559	9.045	1.084	0.783
0.55	0.932	0.75	28.603	1.146	0.776
0.31	0.754	1.542	7.896	1.327	0.735
0.967	0.999	0.1	60.345	1.343	0.733
0.649	0.958	0.529	12.436	1.258	0.748
0.54	0.881	1.082	7.796	0.985	0.803
0.913	0.999	0.179	4.659	1.036	0.793
0.362	0.72	1.931	52.779	1.025	0.795
0.635	0.926	0.59	12.945	1.03	0.794
0.93	0.998	0.16	17.849	1.347	0.73
0.841	0.999	0.245	4.521	0.966	0.807
"""
data['GHORP'] = """
0.563	0.895	0.993	17.544	1.138	0.772
0.948	0.998	0.137	11.678	1.021	0.796
0.759	0.952	0.51	38.459	1.074	0.79
0.647	0.965	0.516	21.704	1.201	0.76
0.898	0.988	0.232	8.955	1.209	0.758
0.793	0.988	0.309	9.518	1.112	0.778
0.677	0.959	0.505	7.33	0.901	0.82
0.915	0.997	0.169	7.701	0.913	0.818
0.512	0.969	0.633	45.074	0.894	0.821
0.679	0.958	0.485	30.626	0.977	0.805
0.923	0.994	0.155	17.351	1.194	0.761
0.759	0.976	0.378	5.44	0.919	0.816
"""


data['0.25'] = """
0.616	0.966	0.537	28.268	1.124	0.775
0.924	0.999	0.163	6.804	1.016	0.797
0.57	0.949	0.691	226.11	1.069	0.791
0.318	0.708	1.591	20.495	1.15	0.77
0.85	0.97	0.373	11.374	1.208	0.759
0.745	0.965	0.402	18.276	1.081	0.784
0.748	0.978	0.402	8.575	0.878	0.824
0.903	0.997	0.191	16.8	0.873	0.825
0.411	0.936	0.865	220.98	0.792	0.842
0.765	0.994	0.371	24.949	0.917	0.816
0.686	0.974	0.443	10.783	1.153	0.769
0.78	0.988	0.316	6.45	0.909	0.818
"""

data['0.75'] = """
0.426	0.697	1.891	26.97	1.121	0.776
0.905	0.999	0.186	11.758	1.027	0.794
0.572	0.942	0.671	27.318	1.089	0.787
0.369	0.62	4.246	30.71	1.186	0.763
0.013	0.093	30.871	368.969	1.262	0.748
0.546	0.807	1.061	11.761	1.152	0.77
0.623	0.88	0.824	8.259	0.903	0.819
0.884	0.997	0.227	6.571	0.946	0.811
0.371	0.809	1.305	32.741	0.913	0.817
0.296	0.533	5.27	27.672	0.95	0.81
0.313	0.873	1.206	14.387	1.237	0.753
0.583	0.864	0.832	9.632	0.927	0.815
"""

data['more_data'] = """
0.539	0.851	1.323	9.277	1.162	0.768
0.854	0.995	0.254	11.823	0.979	0.804
0.404	0.867	1.084	26.082	1.079	0.79
0.367	0.796	1.457	25.758	1.216	0.757
0.868	0.999	0.212	8.464	1.341	0.732
0.82	0.991	0.29	14.506	1.068	0.786
0.646	0.929	0.593	13.935	0.894	0.821
0.826	0.992	0.301	10.699	0.943	0.811
0.304	0.812	1.313	48.419	0.922	0.816
0.694	0.905	0.588	14.973	0.93	0.814
0.894	0.998	0.192	13.186	1.257	0.749
0.637	0.963	0.518	16.67	0.888	0.822
"""

data['more_data_sem'] = """
0.498	0.858	1.13	24.287	1.149	0.77
0.883	0.997	0.2	9.253	0.975	0.805
0.392	0.865	1.098	22.821	1.088	0.788
0.267	0.661	1.968	100.663	1.209	0.758
0.774	0.936	0.485	28.495	1.236	0.753
0.784	0.993	0.315	16.131	1.08	0.784
0.539	0.86	1.096	10.64	0.888	0.823
0.821	0.99	0.308	12.786	0.933	0.813
0.25	0.748	1.453	30.985	0.949	0.81
0.562	0.814	0.937	7.521	0.954	0.809
0.771	0.931	0.472	25.533	1.144	0.771
0.744	0.976	0.36	6.123	0.897	0.821
"""

def parse_table(data):

    # put raw into:
    # {'method': {'category': {'F5': 0.62, 'F10': 0.91, 'CD': 0.77}}}
    table = {}
    for method in data:
        table[method] = {} 
        lines = data[method].split('\n')
        for l, line in enumerate(lines):
            if len(line) == 0:
                continue
            line = line.split()
            category = categories[(l - 1) % len(categories)]
            for i in range(len(line)):
                metric = metrics[i]
                if category not in table[method]:
                    table[method][category] = defaultdict(list)
                table[method][category][metric].append(float(line[i]))
    for method in table:
        for category in table[method]:
            for metric in table[method][category]:
                table[method][category][metric] = np.mean(table[method][category][metric])
    return table


def sort_table(table):
    # {'method': {'category': {'F5': 0.62, 'F10': 0.91, 'CD': 0.77}}}
    # record first num and second num's method, category, metric
    return

def count_mean(table):
    # {'method': {'category': {'F5': 0.62, 'F10': 0.91, 'CD': 0.77}}}
    # make "mean" as category label that calculate the mean of all categories
    for method in table:
        table[method]['mean'] = defaultdict(list)
        for metric in table[method]['Mug']:
            table[method]['mean'][metric] = []
            for category in table[method]:
                if category == 'mean':
                    continue
                table[method]['mean'][metric].append(table[method][category][metric])
            table[method]['mean'][metric] = np.mean(table[method]['mean'][metric])
    return
def print_table(table, method_list, cat_list, metric_list, ignore=0):
    # {'method': {'category': {'F5': 0.62, 'F10': 0.91, 'CD': 0.77}}}
    str_table = []
    # crate str_table of with rows as method_list, and cols as cat_list x metric_list
    str_table.append(['method'] + [cat + '_' + metric for cat in cat_list for metric in metric_list])
    for method in method_list:
        str_table.append([method])
        for cat in cat_list:
            for metric in metric_list:
                if type(table[method][cat][metric]) == list:
                    text = ''
                else:
                    text = f'{table[method][cat][metric]:.2f}' if 'CD' not in metric else f'{table[method][cat][metric]:.1f}'

                str_table[-1].append(text)# f'{table[method][cat][metric]:.2f}')

    for m, metric in enumerate(metric_list):
        for c, cat in enumerate(cat_list):
            # get all methods' metric
            numbers = []
            for method in method_list:
                numbers.append(np.array(table[method][cat][metric]))
            numbers = np.array(numbers)
            if ignore < 0:
                break
            numbers = numbers[ignore:]

            sort_idx = np.argsort(numbers)[::-1]  # 0,1,2? etc
            if 'CD' in metric or 'MJMPE' in metric or metric in ['Volume', 'max depth', 'avg depth', 'disp mean', 'disp std', ]:
                sort_idx = sort_idx[::-1]
            sort_idx = sort_idx+ignore
            # col_idx = m * len(metric_list) + c + 1
            col_idx = c * len(metric_list) + m + 1
            
            first_template = '\\first \\textbf{{{}}}'
            str_table[sort_idx[0]+1][col_idx] = first_template.format(str_table[sort_idx[0]+1][col_idx])
            second_template = '\\second {}'
            str_table[sort_idx[1]+1][col_idx] = second_template.format(str_table[sort_idx[1]+1][col_idx])


    for i, str_row in enumerate(str_table):
        str_table[i] = ' & '.join(str_row)
    str_table = ' \\\\\n'.join(str_table) + ' \\\\'
    print(str_table)
    return str_table
        

grasp_data = {}
grasp_data["GT"] = """6.16	1.32	0.37	2.32	3.24	0.95	0.15"""
grasp_data["GraspTTA"] = """5.25	2.44	0.61	2.89	3.53	1.00	0.23"""
grasp_data["GHORP"] = """7.55	2.42	0.68	2.48	3.30	0.99	0.20"""
grasp_data["GHORP+"] = """11.46	1.84	0.31	0.95	1.63	1.00	0.23"""

grasp_metrics = ['Volume', 'max depth', 'avg depth', 'disp mean', 'disp std', 'ratio', 'area']

grasp_obman = {}
grasp_obman["GT"] = """1.70	0.98	0.74	1.57	2.49	1.00	0.12"""
grasp_obman["GraspTTA"] = """5.56	0.87	0.58	1.54	2.40	1.00	0.18"""
grasp_obman["GHORP"] = """17.40	0.74	0.51	1.85	3.03	0.93	0.25"""
grasp_obman["GHORP+"] = """8.25	0.74	0.57	3.87	4.12	0.82	0.12"""
grasp_obman["GF"] = """43.35	0.79	0.64	1.82	0	1.00	0.09313045"""

grasp_obman["GF*"] = """6.05	0.56	0.44	2.07	2.81	89.40	0.06"""
grasp_obman["Ours*"] = """6.39	0.97	0.70	2.03	2.98	1.00	0.13"""

grasp_metrics = ['Volume', 'max depth', 'avg depth', 'disp mean', 'disp std', 'ratio', 'area' ]
def parse_grasp_table(grasp_data):
    # table[method][metric] = value
    table = {}
    for method in grasp_data:
        table[method] = {}
        table[method]['mean'] = {}
        line = grasp_data[method].split()
        for i in range(len(line)):
            table[method]['mean'][grasp_metrics[i]] = float(line[i])    
    return table

if __name__ == '__main__':
    table = parse_table(data)
    print_table(table, ['HHOR', 'iHOI', 'Ours'], ['Mug','Bottle','Kettle','Bowl','Knife','ToyCar','mean'], ['F5','F10','CD'])

