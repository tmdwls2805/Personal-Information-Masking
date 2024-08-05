def career2_st(x):

    import pandas as pd

    data = pd.read_excel(x, header=None)
    data = data.fillna(0)

    first_data = []
    first_process1_data = []
    first_process2_data = []
    first_process3_data = []
    inter_result = []
    final_result = []

    columns1 = range(1, 31)
    rows = range(0, 10)

    for j in columns1:
        for i in rows:
            if data[i][j] != 0:
                first_data.append(data[i][j])

    for i in range(0, 7):
        if len(first_process1_data) % 2 == 0:
            first_process1_data.append(first_data[i])
        else:
            first_process1_data.append(first_data[i].replace(" ", ''))

    for j in range(7, 24):
        if len(first_process2_data) % 2 == 0:
            first_process2_data.append(first_data[j])
        else:
            first_process2_data.append(first_data[j].replace(" ", ''))

    for k in range(24, 25):
        first_process3_data.append(first_data[k])

    inter_result.extend(first_process1_data)
    inter_result.extend(first_process2_data)
    inter_result.extend(first_process3_data)

    for i in range(0, len(inter_result)):
        inter_result[i] = inter_result[i].replace('\n', '')
        inter_result[i] = inter_result[i].replace('\t', '')
        inter_result[i] = inter_result[i].strip()

    final_result.append(' '.join(inter_result))

    return final_result