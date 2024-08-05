from _parser.normal.security_form2 import security2
from _parser.normal.career_form1 import career1
from _parser.normal.career_form2 import career2
from _parser.normal.career_form3 import career3
from _parser.normal.pp_form1 import pp1
from _parser.normal.pp_form2 import pp2
from _parser.normal.pp_form3 import pp3
from _parser.normal.ldcc_form1 import ldcc1
from _parser.normal.ldcc_form2 import ldcc2

import os

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
all_data = []

file = ['security_form2', 'security_form1', 'security_form3',
        'career_form1', 'career_form2', 'career_form3',
        'pp_form1', 'pp_form2', 'pp_form3',
        'ldcc_form1', 'ldcc_form2']

# 파일이 열려 있으면 실행이 안됨
for (path, dir, files) in os.walk('../_inputData/raw/documents/'): #여기 부분을 x로 바꾸면 됨 (파일이 들어있는 디렉터리 위치)
    print(files)
    for filename in files:
        print(filename)
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
                elif i == 'ldcc_form1':
                    ldcc_form1_path.append(path + '/' + filename)
                elif i == 'ldcc_form2':
                    ldcc_form2_path.append(path + '/' + filename)


for i in range(0, len(security_form1_path)):
    Security1 = security2(security_form1_path[i])
    all_data.append(Security1)

for i in range(0, len(security_form2_path)):
    Security2 = security2(security_form2_path[i])
    all_data.append(Security2)

for i in range(0, len(security_form3_path)):
    Security3 = security2(security_form3_path[i])
    all_data.append(Security3)

for i in range(0, len(career_form1_path)):
    Career1 = career1(career_form1_path[i])
    all_data.append(Career1)

for i in range(0, len(career_form2_path)):
    Career2 = career2(career_form2_path[i])
    all_data.append(Career2)

for i in range(0, len(career_form3_path)):
    Career3 = career3(career_form3_path[i])
    all_data.append(Career3)

for i in range(0, len(pp_form1_path)):
    Pp1 = pp1(pp_form1_path[i])
    all_data.append(Pp1)

for i in range(0, len(pp_form2_path)):
    Pp2 = pp2(pp_form2_path[i])
    all_data.append(Pp2)

for i in range(0, len(pp_form3_path)):
    Pp3 = pp3(pp_form3_path[i])
    all_data.append(Pp3)

for i in range(0, len(ldcc_form1_path)):
    Ldcc1 = ldcc1(ldcc_form1_path[i])
    all_data.append(Ldcc1)

for i in range(0, len(ldcc_form2_path)):
    Ldcc2 = ldcc2(ldcc_form2_path[i])
    for i in range(0, len(Ldcc2)):
        all_data.append(Ldcc2[i])

for i in range(0, len(all_data)):
    for j in range(0, len(all_data[i])):
        all_data[i][j] = all_data[i][j].replace('  ', ' ')

print(all_data)
print(len(all_data))

f = open('../all_data_normal.txt', mode='wt', encoding='utf-8')

for i in range(0, len(all_data)):
    for j in range(0, len(all_data[i])):
        f.write(str(all_data[i][j]))
        f.write('\n')
    f.write('\n')
