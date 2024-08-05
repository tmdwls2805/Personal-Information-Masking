from _dataGenerator.Generation_careerFIle import create_careerFIle_data
form, num = input("예시)form1 100 : ").split()
num = int(num)
start = create_careerFIle_data.create(num)
if form == "form1":
    start.form1()
elif form == "form2":
    start.form2()
elif form == "form3":
    start.form3()