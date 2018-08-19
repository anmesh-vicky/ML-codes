#16NA30003
#Anmesh Choudhury
#Assigment Number 1
#To run python <filename> <dataset>

import csv
import sys 

rows=[]
with open(sys.argv[1],'r')as csvfile:
	csvreader=csv.reader(csvfile)
	for row  in csvreader:
		rows.append(row)
hypothesis=['-2']*8 #declaring the most general hypothesis 
for i in range(len(rows)):
	if (rows[i][8]=='1'):
		for z in range(8):
			if hypothesis[z]=='-2':
				hypothesis[z]=rows[i][z]
			elif hypothesis[z]!=rows[i][z]:
				hypothesis[z]=-1
		
c=0
for i in range(len(hypothesis)):
	if(hypothesis[i]=='1'):
		c+=1
	elif(hypothesis[i]=='0'):
		c+=1

print c,
for i in range(len(hypothesis)):
	if(hypothesis[i]=='1'):
		print i+1,
	elif(hypothesis[i]=='0'):
		print (i+1)*-1,






