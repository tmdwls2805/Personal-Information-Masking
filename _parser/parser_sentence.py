from _parser.sentence.security_form_st import security_st
from _parser.sentence.introduce_form1_st import introduce1_st
from _parser.sentence.introduce_form2_st import introduce2_st
from _parser.sentence.introduce_form3_st import introduce3_st
from _parser.sentence.career_form1_st import career1_st
from _parser.sentence.career_form2_st import career2_st
from _parser.sentence.career_form3_st import career3_st
from _parser.sentence.pp_form1_st import pp1_st
from _parser.sentence.pp_form2_st import pp2_st
from _parser.sentence.pp_form3_st import pp3_st
from _parser.sentence.ldcc_form1_st import ldcc1_st
from _parser.sentence.ldcc_form2_st import ldcc2_st
import os

def pred_data_parsing(absolute_path, file_name):
    security_form1_path = []
    security_form2_path = []
    security_form3_path = []
    career_form1_path = []
    career_form2_path = []
    career_form3_path = []
    pp_form1_path = []
    pp_form2_path = []
    pp_form3_path = []
    ldcc_form1_path = []
    ldcc_form2_path = []
    introduce_form1_path = []
    introduce_form2_path = []
    introduce_form3_path = []
    all_data = []

    file = ['security_form2', 'security_form1', 'security_form3',
            'career_form1', 'career_form2', 'career_form3',
            'pp_form1', 'pp_form2', 'pp_form3',
            'introduce_form1', 'introduce_form2', '자기소개서',
            'ldcc_form1', 'ldcc_form2']

    # 파일이 열려 있으면 실행이 안됨
    files = []
    path = absolute_path + "_inputData/raw/documents"
    files.append(file_name)
    for filename in files:
        for i in file:
            if i in filename:
                # print('%s파일에 %s가 포함' %(filename, i))
                if i == 'security_form1':
                    security_form1_path.append(path+'/'+filename)
                elif i == 'security_form2':
                    security_form2_path.append(path + '/' + filename)
                elif i == 'security_form3':
                    security_form3_path.append(path + '/' + filename)
                elif i == 'career_form1':
                    career_form1_path.append(path + '/' + filename)
                elif i == 'career_form2':
                    career_form2_path.append(path + '/' + filename)
                elif i == 'career_form3':
                    career_form3_path.append(path + '/' + filename)
                elif i == 'pp_form1':
                    pp_form1_path.append(path + '/' + filename)
                elif i == 'pp_form2':
                    pp_form2_path.append(path + '/' + filename)
                elif i == 'pp_form3':
                    pp_form3_path.append(path + '/' + filename)
                elif i == 'introduce_form1':
                    introduce_form1_path.append(path + '/' + filename)
                elif i == 'introduce_form2':
                    introduce_form2_path.append(path + '/' + filename)
                elif i == '자기소개서':
                    introduce_form3_path.append(path + '/' + filename)
                elif i == 'ldcc_form1':
                    ldcc_form1_path.append(path + '/' + filename)
                elif i == 'ldcc_form2':
                    ldcc_form2_path.append(path + '/' + filename)


    for i in range(0, len(security_form1_path)):
        Security1 = security_st(security_form1_path[i])
        all_data.append(Security1)

    for i in range(0, len(security_form2_path)):
        Security2 = security_st(security_form2_path[i])
        all_data.append(Security2)

    for i in range(0, len(security_form3_path)):
        Security3 = security_st(security_form3_path[i])
        all_data.append(Security3)

    for i in range(0, len(career_form1_path)):
        Career1 = career1_st(career_form1_path[i])
        all_data.append(Career1)

    for i in range(0, len(career_form2_path)):
        Career2 = career2_st(career_form2_path[i])
        all_data.append(Career2)

    for i in range(0, len(career_form3_path)):
        Career3 = career3_st(career_form3_path[i])
        all_data.append(Career3)

    for i in range(0, len(pp_form1_path)):
        Pp1 = pp1_st(pp_form1_path[i])
        all_data.append(Pp1)

    for i in range(0, len(pp_form2_path)):
        Pp2 = pp2_st(pp_form2_path[i])
        all_data.append(Pp2)

    for i in range(0, len(pp_form3_path)):
        Pp3 = pp3_st(pp_form3_path[i])
        all_data.append(Pp3)

    for i in range(0, len(introduce_form1_path)):
        Introduce1 = introduce1_st(introduce_form1_path[i])
        all_data.append(Introduce1)

    for i in range(0, len(introduce_form2_path)):
        Introduce2 = introduce2_st(introduce_form2_path[i])
        all_data.append(Introduce2)

    for i in range(0, len(introduce_form3_path)):
        Introduce3 = introduce3_st(introduce_form3_path[i])
        all_data.append(Introduce3)

    for i in range(0, len(ldcc_form1_path)):
        Ldcc1 = ldcc1_st(ldcc_form1_path[i])
        all_data.append(Ldcc1)

    for i in range(0, len(ldcc_form2_path)):
        Ldcc2 = ldcc2_st(ldcc_form2_path[i])
        all_data.append(Ldcc2)

    for i in range(0, len(all_data)):
        for j in range(0, len(all_data[i])):
            all_data[i][j] = all_data[i][j].replace('  ', ' ')

    # print(all_data)
    # print(len(all_data))

    # f = open('../all_data_sentence.txt', mode='wt', encoding='utf-8')
    #
    # for i in range(0, len(all_data)):
    #     for j in range(0, len(all_data[i])):
    #         f.write(str(all_data[i][j]))
    #         f.write('\n')

    return all_data