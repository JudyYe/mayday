import numpy as np

metrics = ["ADD", "ADD-S", "Center", "ACC-NORM"]
data = {}

category = ["All", "Contact", "Truncated", "Out-of-view"]
data['Obj'] = """
	ADD	ADD-S	Center	ACC-NORM	ADD	ADD-S	Center	ADD	ADD-S	Center	ADD	ADD-S	Center
FP+HAWOR-simple	0.301	0.447	0.367	0.896	0.387	0.575	0.481	0.194	0.289	0.232	0.083	0.137	0.099
FP+HAWOR-contact	0.309	0.458	0.375	0.627	0.396	0.586	0.490	0.201	0.333	0.249	0.088	0.169	0.104
Ours	0.511	0.699	0.580	0.110	0.558	0.773	0.650	0.425	0.602	0.490	0.288	0.438	0.305
"""

data["HOI"] = """
	ADD	ADD-S	Center	ACC-NORM	ADD	ADD-S	Center	ADD	ADD-S	Center	ADD	ADD-S	Center
FP+HAWOR-simple	0.300	0.451	0.365	0.896	0.379	0.571	0.471	0.183	0.298	0.220	0.088	0.172	0.104
FP+HAWOR-contact	0.389	0.568	0.482	0.627	0.496	0.719	0.624	0.284	0.480	0.386	0.084	0.201	0.096
Ours	0.535	0.723	0.608	0.110	0.580	0.794	0.676	0.443	0.627	0.517	0.280	0.432	0.295
"""

hand_metrics = ["PA-MPJPE", "W-MPJPE", "WA-MPJPE", "Accel"]
category_hand = ["All"]
data["Hand"] = """
		PA-MPJPE	W-MPJPE	WA-MPJPE	Accel
HaWoR		0.00899	0.11256	0.0376 0.037594	0.04148
FP+HAWOR-simple		0.00899	0.10675	0.03926	0.01094
FP+HAWOR-contact		0.00899	0.18894	0.07903	0.01520
Ours		0.00667	0.10412	0.03263	0.00577
"""


category_ablation = ["Hand", "Object", "Interaction"]
data["Ablation"] = """
		PA-MPJPE	W-MPJPE 	WA-MPJPE	Accel	ADD-obj	ADDI-obj	Center-obj	ACC-obj	ADD-hoi	ADDI-hoi	Center-hoi	ACC-hoi
ours		0.00667	0.10412	0.03263	0.00577	0.511	0.699	0.580	0.110	0.535	0.723	0.608	0.110
GT-contact	0.006612	0.107923	0.032779	0.005518	0.537	0.7217142222	0.6047577778	0.07040227205	0.561	0.745	0.6313971111	0.07040216029
no-guidance		0.009345	0.159229	0.04595	0.007592	0.443	0.6104111111	0.50978	0.1332766414	0.450	0.6212884444	0.5224046667	0.1332766414
no-contact		0.006703	0.104701	0.031221	0.006099	0.285	0.4294973333	0.3207328889	0.08681160212	0.306	0.4547377778	0.3449206667	0.08681152016
"""
def parse_table(data, metrics, category):
    """
    Parse metrics text for an object table.

    Args:
        data: dict containing the key "Obj" with tab-separated values.

    Returns:
        dict mapping method -> category -> metric -> value
    """
    raw = data.strip().splitlines()
    header = raw[0].split()

    subset_names = category
    subset_columns = []
    idx = 0
    for subset in subset_names:
        metrics_available = []
        for metric in metrics:
            if idx < len(header) and header[idx].lower() == metric.lower():
                metrics_available.append(metric)
                idx += 1
            else:
                break
        subset_columns.append((subset, metrics_available))

    table = {}
    for row in raw[1:]:
        row_split = row.split()
        method = row_split[0]
        table[method] = {}
        cursor = 1
        for subset, metric_list in subset_columns:
            table[method][subset] = {}
            for metric in metric_list:
                value = float(row_split[cursor])
                table[method][subset][metric] = value
                cursor += 1

    return table


def print_table(table, method_list, cat_list, metric_list, format_dict=None, no_first=False):
    # {'method': {'category': {'F5': 0.62, 'F10': 0.91, 'CD': 0.77}}}
    # print string ready for latex
    # method & ADD & ADD-S & Center & ACC-NORM & ADD & ADD-S & Center & ADD & ADD-S & Center & ADD & ADD-S & Center \\
    # FP+HAWOR-simple & 0.301 & 0.447 & 0.367 & 0.896 & .... \\
    # FP+HAWOR-contact & 0.309 & 0.458 & 0.375 & 0.627 \\
    # Ours & \first \textbf{0.511} & \first \textbf{0.699} & \first \textbf{0.580} & \first \textbf{0.110} \\

    if len(metric_list) != len(cat_list):
        print("????" ,len(metric_list), len(cat_list))

    normalized_metric_list = []
    for metrics_for_cat in metric_list:
        if isinstance(metrics_for_cat, (list, tuple)):
            normalized_metric_list.append(list(metrics_for_cat))
        else:
            normalized_metric_list.append([metrics_for_cat])
    metric_list = normalized_metric_list

    header = ["method"]
    for cat, metrics_for_cat in zip(cat_list, metric_list):
        for metric in metrics_for_cat:
            header.append(f"{metric}")

    header_rows = []
    first_header = ["method"]
    
    for cat, metrics_for_cat in zip(cat_list, metric_list):
        count = len(metrics_for_cat)
        first_header.append(f"\\multicolumn{{{count}}}{{c}}{{{cat}}}")
    header_rows.append(" & ".join(first_header))

    second_header = [" "]
    current_col = 2
    rules = []
    for metrics_for_cat in metric_list:
        count = len(metrics_for_cat)
        second_header.extend(metrics_for_cat)
        start = current_col
        end = current_col + count - 1
        rules.append(f"\\cmidrule(r){{{start}-{end}}}")
        current_col += count
    header_rows.append(" ".join(rules))
    header_rows.append(" & ".join(second_header))
    print((" \\\\\n".join(header_rows)) + " \\\\\n\\midrule")

    # rows = [["method"] + [metric for metrics_for_cat in metric_list for metric in metrics_for_cat]]
    rows = []
    for method in method_list:
        row = [method]
        for cat, metrics_for_cat in zip(cat_list, metric_list):
            for metric in metrics_for_cat:
                value = table[method][cat].get(metric, np.nan)
                if isinstance(value, float) and np.isnan(value):
                    text = ""
                else:
                    if format_dict and metric in format_dict:
                        text = format_dict[metric](value)
                    # else:
                    #     text = f"{value:.3f}"
                row.append(text)
        rows.append(row)

    frame_rows = []
    for row_idx, row in enumerate(rows):
        frame_rows.append(row)

    num_metrics_total = sum(len(metrics_for_cat) for metrics_for_cat in metric_list)

    for metric_offset in range(num_metrics_total):
        column_values = []
        for row in frame_rows:
            value_str = row[metric_offset + 1]
            if value_str == "" or value_str.startswith("\\first"):
                column_values.append(np.nan)
            else:
                column_values.append(float(value_str))
        column_values = np.array(column_values)
        metric_name = None
        count_tracker = 0
        for metrics_for_cat in metric_list:
            if metric_offset < count_tracker + len(metrics_for_cat):
                metric_name = metrics_for_cat[metric_offset - count_tracker]
                break
            count_tracker += len(metrics_for_cat)

        if metric_name in ["ACC-NORM", "Accel"] or "MPJPE" in metric_name:
            best_idx = np.nanargmin(column_values)
        else:
            best_idx = np.nanargmax(column_values)
        best_value = column_values[best_idx]
        # best_value = format_dict[metric_name](best_value)

        if not no_first:
            frame_rows[best_idx][metric_offset + 1] = f"\\first \\textbf{{{best_value}}}"

    table_str = " \\\\\n".join([" & ".join(row) for row in frame_rows]) + " \\\\"
    print(table_str)
    return table_str
        