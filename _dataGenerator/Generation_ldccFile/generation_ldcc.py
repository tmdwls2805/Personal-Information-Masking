from _dataGenerator.Generation_ldccFile import create_ldccFIle_data

form, num = input("예시)form1 100 : ").split()
num = int(num)
start = create_ldccFIle_data.create(num)
if form == "form1":
    start.form1()
elif form == "form2":
    start.form2()