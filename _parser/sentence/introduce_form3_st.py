def introduce3_st(x):
    from docx import Document
    import kss

    final_result = []
    doc = Document(x)
    result = [p.text for p in doc.paragraphs]
    inter_result = [v for v in result if v]

    for i in range(0, len(inter_result)):
        for sent in kss.split_sentences(inter_result[i]):
            final_result.append(sent)

    return final_result