def introduce1_st(x):
    import pandas as pd
    import kss

    data = pd.read_excel(x, header=None)
    data = data.fillna(0)

    all_data = []
    inter_result = []
    final_result = []
    columns = range(1, 7)
    columns2 = range(7, 35)
    rows2 = range(1, 9)
    rows = range(9)

    for j in columns:
        for i in rows:
            if data[i][j] != 0:
                all_data.append(data[i][j])

    for j in columns2:
        for i in rows2:
            if data[i][j] != 0:
                all_data.append(data[i][j])

    for i in range(0, 6):
        inter_result.append(all_data[i])

    final_result.append(' '.join(inter_result))

    for i in range(6, len(all_data)):
        for sent in kss.split_sentences(all_data[i]):
            final_result.append(sent)

    return final_result