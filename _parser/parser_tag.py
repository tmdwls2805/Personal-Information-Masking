from _parser.tag.career_form1_tag import career1_tag
from _parser.tag.career_form2_tag import career2_tag
from _parser.tag.career_form3_tag import career3_tag
from _parser.tag.pp_form1_tag import pp1_tag
from _parser.tag.pp_form2_tag import pp2_tag
from _parser.tag.pp_form3_tag import pp3_tag
from _parser.tag.security_form1_tag import security1_tag
from _parser.tag.security_form2_tag import security2_tag
from _parser.tag.security_form3_tag import security3_tag
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
all_data = []

file = ['security_form2', 'security_form1', 'security_form3',
        'career_form1', 'career_form2', 'career_form3',
        'pp_form1', 'pp_form2', 'pp_form3']

# 파일이 열려 있으면 실행이 안됨
for (path, dir, files) in os.walk('../_inputData/raw/documents/'): #여기 부분을 x로 바꾸면 됨 (파일이 들어있는 디렉터리 위치)
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


for i in range(0, len(security_form1_path)):
    Security1 = security1_tag(security_form1_path[i])
    all_data.append(Security1)
    print("security_form1_%s번째" % i)

for i in range(0, len(security_form2_path)):
    Security2 = security2_tag(security_form2_path[i])
    all_data.append(Security2)
    print("security_form2_%s번째" % i)

for i in range(0, len(security_form3_path)):
    print("security_form3_%s번째" %i)
    Security3 = security3_tag(security_form3_path[i])
    all_data.append(Security3)

for i in range(0, len(career_form1_path)):
    print("career_form1_%s번째" % i)
    Career1 = career1_tag(career_form1_path[i])
    all_data.append(Career1)

for i in range(0, len(career_form2_path)):
    print("career_form2_%s번째" % i)
    Career2 = career2_tag(career_form2_path[i])
    all_data.append(Career2)

for i in range(0, len(career_form3_path)):
    print("career_form3_%s번째" % i)
    Career3 = career3_tag(career_form3_path[i])
    all_data.append(Career3)

for i in range(0, len(pp_form1_path)):
    print("pp_form1_%s번째" % i)
    Pp1 = pp1_tag(pp_form1_path[i])
    all_data.append(Pp1)

for i in range(0, len(pp_form2_path)):
    print("pp_form2_%s번째" % i)
    Pp2 = pp2_tag(pp_form2_path[i])
    all_data.append(Pp2)

for i in range(0, len(pp_form3_path)):
    print("pp_form3_%s번째" % i)
    Pp3 = pp3_tag(pp_form3_path[i])
    all_data.append(Pp3)

for i in range(0, len(all_data)):
    for j in range(0, len(all_data[i])):
        all_data[i][j] = all_data[i][j].replace('  ', ' ')

print(all_data)
print(len(all_data))

f = open('../all_data_tag.txt', mode='wt', encoding='utf-8')


for i in range(0, len(all_data)):
    for j in range(0, len(all_data[i])):
        f.write(str(all_data[i][j]))
        f.write('\n')
    f.write('\n')
