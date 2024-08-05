def career3_st(x):
    import pandas as pd

    data = pd.read_excel(x, header=None)
    data = data.fillna(0)

    inter_result = []
    final_result = []
    first_data = []
    first_process1_data = []
    first_process1_1_data = []
    first_process2_data = []
    first_process2_1_data = []
    first_process3_data = []
    first_process3_1_data = []

    columns = range(8, 35)

    rows = range(0, 10)

    for j in columns:
        for i in rows:
            if data[i][j] != 0:
                first_data.append(data[i][j])

    for i in range(0, 7):
        if len(first_process1_data) % 2 == 0:
            first_process1_data.append(first_data[i])
        else:
            first_process1_data.append(first_data[i].replace(" ", ''))

    for i in range(0, len(first_process1_data)):
        if len(first_process1_1_data) == 2:
            first_process1_1_data.append(first_process1_data[i].replace(" ", ''))
        else:
            first_process1_1_data.append(first_process1_data[i])

    for i in range(7, 30):
        if len(first_process2_data) % 2 == 0:
            first_process2_data.append(first_data[i])
        else:
            first_process2_data.append(first_data[i].replace(" ", ''))

    for i in range(0, len(first_process2_data)):
        if len(first_process2_1_data) == 6:
            first_process2_1_data.append(first_process2_data[i].replace(" ", ''))
        elif len(first_process2_1_data) == 14:
            first_process2_1_data.append(first_process2_data[i].replace(" ", ''))
        else:
            first_process2_1_data.append(first_process2_data[i])

    for i in range(30, 39):
        if len(first_process3_data) % 2 == 0:
            first_process3_data.append(first_data[i])
        else:
            first_process3_data.append(first_data[i].replace(" ", ''))

    for i in range(0, len(first_process3_data)):
        if len(first_process3_1_data) == 2:
            first_process3_1_data.append(first_process3_data[i].replace(" ", ''))
        elif len(first_process3_1_data) == 4:
            first_process3_1_data.append(first_process3_data[i].replace(" ", ''))
        else:
            first_process3_1_data.append(first_process3_data[i])

    inter_result.extend(first_process1_1_data)
    inter_result.extend(first_process2_1_data)
    inter_result.extend(first_process3_1_data)
    inter_result.append(first_data[39])

    for i in range(0, len(inter_result)):
        inter_result[i] = inter_result[i].replace('\n', '')
        inter_result[i] = inter_result[i].replace('\t', '')
        inter_result[i] = inter_result[i].strip()

    final_result.append(' '.join(inter_result))

    return final_result