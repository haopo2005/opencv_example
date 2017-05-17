# -*- coding:gbk -*-  
import os
import sys
import os.path
posdir = sys.argv[1] 
imgdir = sys.argv[2]                                  


f=open('/home/jst/share/data.info', 'w')

list_img = os.listdir(imgdir)
list = sorted(list_img)
for j in range(0,len(list)):
	path = os.path.join(imgdir,list[j])
	f.write(path+'\r\n')
f.close()

list_pos = os.listdir(posdir)
list = sorted(list_pos)
number=[]
for i in range(0,len(list)):
	path = os.path.join(posdir,list[i])
	print "filename is:" + path
	a = 0
	for line in open(path):
		if a == 0:
			a = 1
			continue
		#print repr(line)
		#fuck!
		#'256\t110\t201\t104\r\n'
		#'218\t64\t173\t65\r\n'
		#'208\t95\t157\t93\r\n'
		number_list = line.split('\t')
		number_list[3] = filter(str.isdigit, number_list[3])
		number.append(number_list)
		#print number_list
#print number

index=0
width=40
height=20
ff=open('/home/jst/share/TrainEye/eye.info', 'w')
for line in open('/home/jst/share/data.info', 'r'):
	#print line
	l_x=int(number[index][0])
	l_y=int(number[index][1])
	r_x=int(number[index][2])
	r_y=int(number[index][3])
	index=index+1
	l_rect_x=l_x-width/2
	l_rect_y=l_y-height/2
	r_rect_x=r_x-width/2
	r_rect_y=r_y-height/2
	ss = line.split('\r\n')
	xx="%s 2 %d %d %d %d %d %d %d %d\r\n" % (ss[0], l_rect_x,l_rect_y,width,height,r_rect_x,r_rect_y,width,height)
	ff.write(xx)
ff.close()
