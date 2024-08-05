import hgtk

EMPTY_JS_CHAR = "#"

def jamo_split(sentence):
    result = []
    for word in sentence:
        decomposed_word = ""
        for char in word:
            try:
                cho_joong_jong = hgtk.letter.decompose(char)
                char_seq = ""
                for cvc in cho_joong_jong:
                    if cvc == '':
                        cvc = EMPTY_JS_CHAR
                    char_seq += cvc
                decomposed_word += char_seq
            except hgtk.exception.NotHangulException:
                decomposed_word += char
                continue
        result.append(decomposed_word)
    return " ".join(result)