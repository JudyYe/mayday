import numpy as np
raw = """ Ours			iHOI			HHOR		
	F5	F10	CD	F5	F10	CD	F5	F10	CD
Mug	0.64	0.86	1.01	0.44	0.71	2.10	0.18	0.37	6.95
Bottle	0.54	0.92	0.72	0.48	0.77	1.48	0.26	0.56	3.09
Kettle	0.43	0.77	1.50	0.21	0.45	6.31	0.12	0.30	11.35
Bowl	0.79	0.98	0.36	0.38	0.64	3.10	0.31	0.54	4.18
Knife	0.50	0.95	0.76	0.33	0.68	2.82	0.71	0.93	0.63
ToyCar	0.83	0.99	0.27	0.66	0.95	0.54	0.26	0.59	1.90
mean	0.62	0.91	0.77	0.42	0.70	2.73	0.31	0.55	4.68
"""

def parse_table(raw):

    # put raw into:
    # {'method': {'category': {'F5': 0.62, 'F10': 0.91, 'CD': 0.77}}}
    table = {}
    lines = raw.split('\n')
    num_metric = 0
    for l, line in enumerate(lines):
        # if l == 0:
        #     continue
        if len(line) == 0:
            continue
        if l == 0:
            methods = line.split()
            for method in methods:
                table[method] = {} 
            continue
        if l == 1:
            metrics = line.split()
            num_metric = len(set(metrics))
            continue
        line = line.split()
        category = line[0]
        for i in range(1, len(line)):
            # if len(line[i]) == 0:
            #     continue
            category = line[0]
            method = methods[(i - 1) // num_metric]
            if category not in table[method]:
                table[method][category] = {}
            metric = metrics[i-1]
            table[method][category][metric] = float(line[i])

    return table

def sort_table(table):
    # {'method': {'category': {'F5': 0.62, 'F10': 0.91, 'CD': 0.77}}}
    # record first num and second num's method, category, metric
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
                text = f'{table[method][cat][metric]:.2f}' if 'CD' not in metric else f'{table[method][cat][metric]:.1f}'
                str_table[-1].append(text)# f'{table[method][cat][metric]:.2f}')

    for m, metric in enumerate(metric_list):
        for c, cat in enumerate(cat_list):
            # get all methods' metric
            numbers = []
            for method in method_list:
                numbers.append(np.array(table[method][cat][metric]))
            numbers = np.array(numbers)
            numbers = numbers[ignore:]

            sort_idx = np.argsort(numbers)[::-1]  # 0,1,2? etc
            if 'CD' in metric:
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
        

if __name__ == '__main__':
    table = parse_table(raw)
    print_table(table, ['HHOR', 'iHOI', 'Ours'], ['Mug','Bottle','Kettle','Bowl','Knife','ToyCar','mean'], ['F5','F10','CD'])

