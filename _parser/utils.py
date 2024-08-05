def sel_sort(a):
    n = len(a)
    for i in range(0, n - 1):
        min_idx = i
        for j in range(i + 1, n):
            if a[j][0] < a[min_idx][0]:
                min_idx = j
        a[i], a[min_idx] = a[min_idx], a[i]
    return a


def replaceRight(original, old, new, count_right):
    repeat = 0
    text = original
    old_len = len(old)

    count_find = original.count(old)
    if count_right > count_find:  # 바꿀 횟수가 문자열에 포함된 old보다 많다면
        repeat = count_find  # 문자열에 포함된 old의 모든 개수(count_find)만큼 교체한다
    else:
        repeat = count_right  # 아니라면 입력받은 개수(count)만큼 교체한다

    while (repeat):
        find_index = text.rfind(old)  # 오른쪽부터 index를 찾기위해 rfind 사용
        text = text[:find_index] + new + text[find_index + old_len:]

        repeat -= 1
    return text


def ngram(s, num):
    res = []
    slen = len(s) - num + 1
    for i in range(slen):
        ss = s[i:i+num]
        res.append(ss)
    return res