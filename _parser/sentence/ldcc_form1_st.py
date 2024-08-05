def ldcc1_st(x):
    import pandas as pd

    data = pd.read_excel(x, header=None)
    data = data.fillna(0)

    first_data = []
    second_data = []
    third_data = []
    inter_result = []
    final_result = []

    columns1 = range(0, 14)
    columns2 = range(15, 21)
    columns3 = range(22, 30)
    rows1 = range(0, 7)
    rows2 = range(1, 6)
    for j in columns1:
        for i in rows1:
            if data[i][j] != 0:
                first_data.append(str(data[i][j]))

    for i in columns2:
        for j in rows2:
            if data[j][i] != 0:
                if data[j][14] != 0:
                    second_data.append(str(data[j][14]))
                    second_data.append(str(data[j][i]))
                else:
                    second_data.append(str(data[j][i]))

    for i in columns3:
        for j in rows1:
            if data[j][i] != 0:
                third_data.append(str(data[j][i]))

    for i in range(0, len(first_data)):
        inter_result.append(first_data[i])
    inter_result.append(str(data[0][14]))
    for i in range(0, len(second_data)):
        inter_result.append(second_data[i])
    for i in range(0, len(third_data)):
        inter_result.append(third_data[i])

    for i in range(0, len(inter_result)):
        inter_result[i] = inter_result[i].replace('\n', '')
        inter_result[i] = inter_result[i].replace('\t', '')
        inter_result[i] = inter_result[i].strip()

    final_result.append(' '.join(inter_result))

    return final_result