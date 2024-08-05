def security_st(x):
    from docx import Document

    doc = Document(x)
    result = [p.text for p in doc.paragraphs]
    inter_result = [v for v in result if v]

    final_result = []

    for i in range(1, len(inter_result)):
        final_result.append(inter_result[i])

    for i in range(0, len(final_result)):
        final_result[i] = final_result[i].replace('\n', '')
        final_result[i] = final_result[i].replace('\t', '')
        final_result[i] = final_result[i].strip()

    return final_result