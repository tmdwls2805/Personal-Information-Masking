import numpy as np


def testData_POS(i, model, X_test_sents, X_test_pos, X_test_chars, max_len, max_len_char, y_test2, index_to_word, index_to_ner):
    this_pred = model.predict([X_test_sents, X_test_pos, np.array(X_test_chars).reshape((len(X_test_chars), max_len, max_len_char*3))])
    y_pred = np.argmax(this_pred[i], axis=-1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.
    true = np.argmax(y_test2[i], -1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.

    print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
    print(35 * "-")
    for w, t, pred in zip(X_test_sents[i], true, y_pred):
        if w != 0:  # PAD값은 제외함.
            print("{:17}: {:7} {}".format(index_to_word[w], index_to_ner[t], index_to_ner[pred]))

def testData_POS2(i, model, X_test_sents, X_test_pos, X_test_chars, X_test_jamos, max_len, max_len_char, y_test2, index_to_word, index_to_ner):

    this_pred = model.predict([X_test_sents, X_test_pos, np.array(X_test_chars).reshape((len(X_test_chars), max_len, max_len_char)), np.array(X_test_jamos).reshape((len(X_test_jamos),
                                                                                          max_len, max_len_char*3))])
    y_pred = np.argmax(this_pred[i], axis=-1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.
    true = np.argmax(y_test2[i], -1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.
    print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
    print(35 * "-")
    for w, t, pred in zip(X_test_sents[i], true, y_pred):
        if w != 0:  # PAD값은 제외함.
            print("{:17}: {:7} {}".format(index_to_word[w], index_to_ner[t], index_to_ner[pred]))


def testData_noPOS(i, model, X_test_sents, X_test_chars, max_len, max_len_char, y_test2, index_to_word, index_to_ner):
    this_pred = model.predict([X_test_sents, np.array(X_test_chars).reshape((len(X_test_chars), max_len, max_len_char*3))])
    y_pred = np.argmax(this_pred[i], axis=-1)  # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.
    true = np.argmax(y_test2[i], -1)  # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.

    print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
    print(35 * "-")
    for w, t, pred in zip(X_test_sents[i], true, y_pred):
        if w != 0:  # PAD값은 제외함.
            print("{:17}: {:7} {}".format(index_to_word[w], index_to_ner[t], index_to_ner[pred]))


def sequences_to_tag(sequences, index_to_ner):  # 예측값을 index_to_tag를 사용하여 태깅 정보로 변경하는 함수.
    result = []
    for sequence in sequences:  # 전체 시퀀스로부터 시퀀스를 하나씩 꺼낸다.
        temp = []
        for pred in sequence:  # 시퀀스로부터 예측값을 하나씩 꺼낸다.
            pred_index = np.argmax(pred)  # 예를 들어 [0, 0, 1, 0 ,0]라면 1의 인덱스인 2를 리턴한다.
            temp.append(index_to_ner[pred_index].replace("PAD", "O"))  # 'PAD'는 'O'로 변경
        result.append(temp)
    return result