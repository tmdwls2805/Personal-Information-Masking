def introduce2_st(x):
    import pandas as pd
    import kss

    data = pd.read_excel(x, header=None)
    data = data.fillna(0)

    cnum = [8, 16, 24, 32]
    all_data = []
    inter_result = []
    final_result = []
    columns = range(5, 6)
    rows = range(9)

    for j in columns:
        for i in rows:
            if data[i][j] != 0:
                all_data.append(data[i][j])

    for i in range(0, len(cnum)):
        all_data.append(data[0][cnum[i]])

    for i in range(0, 6):
        inter_result.append(all_data[i])

    final_result.append(' '.join(inter_result))

    for i in range(6, len(all_data)):
        for sent in kss.split_sentences(all_data[i]):
            final_result.append(sent)

    return final_result