import time
import Image
import os
import math

from random import randint
 
CLUSTER_MEAN=[]
NEW_CLUSTER_MEAN=[]
CLUSTER_NO=[]
SN=1
 
##      functions         ##
 
def CLUS(color,K):
    i=1
    dist=math.sqrt( (CLUSTER_MEAN[0][0]-color[0])**2 + (CLUSTER_MEAN[0][1]-color[1])**2 + (CLUSTER_MEAN[0][2]-color[2])**2 )
    clus_num=0
    while i < K:
        temp = math.sqrt( (CLUSTER_MEAN[i][0]-color[0])**2 + (CLUSTER_MEAN[i][1]-color[1])**2 + (CLUSTER_MEAN[i][2]-color[2])**2 )
        if temp < dist :
            dist=temp
            clus_num=i
        i=i+1
    return clus_num
 
def NEW_MEAN(s,K,im_arr):
    count=[]
    color_value=[]
    i=0
    while i < K:
        count.append(0)
        color_value.append([0,0,0])
        i+=1
    for item in CLUSTER_NO :
        count[item[2]]+=1
        color_value[item[2]][0]+=im_arr[item[0],item[1]][0]
        color_value[item[2]][1]+=im_arr[item[0],item[1]][1]
        color_value[item[2]][2]+=im_arr[item[0],item[1]][2]
    i=0
    while i < K:
        value=[0,0,0]
        if count[i]==0:
            value[0]=color_value[i][0]
            value[1]=color_value[i][1]
            value[2]=color_value[i][2]
        else:
            value[0]=color_value[i][0]/count[i]
            value[1]=color_value[i][1]/count[i]
            value[2]=color_value[i][2]/count[i]
        NEW_CLUSTER_MEAN.append(value)
        i+=1
    i=0
    return
 
def MEAN_MOVE(K):
    return math.sqrt( (CLUSTER_MEAN[K][0]-NEW_CLUSTER_MEAN[K][0])**2+ (CLUSTER_MEAN[K][1]-NEW_CLUSTER_MEAN[K][1])**2+ (CLUSTER_MEAN[K][2]-NEW_CLUSTER_MEAN[K][2])**2 )
 
def swap(K) :
    templist=[]
    i=0
    while i< K:
            templist.append(CLUSTER_MEAN[i])
            i+=1
    i=0
    while i< K:
            CLUSTER_MEAN[i]=NEW_CLUSTER_MEAN[i]
            i+=1
    return
 
##      main starts here  ##
 
path=os.getcwd()
dirList=os.listdir(path+"/train")

for k_nick in [3, 5, 10, 25]:
  newpath = path+"/OUTPUT_"+str(k_nick) 
  if not os.path.exists(newpath):
    os.makedirs(newpath)

  for infile in dirList:
      s_time=time.time()
      im=Image.open(path+"/INPUT/"+infile)
      k=k_nick
      f, e = os.path.splitext(infile)
      ARR=im.load()
      S=im.size
      i=0
      while i< k:
	  value=[0,0,0]
	  value[0]=randint(0,255)
	  value[1]=randint(0,255)
	  value[2]=randint(0,255)
	  i=i+1
	  CLUSTER_MEAN.append(value)
      while 1:
	  x=0
	  y=0
	  #finding the initial cluster classification of every pixel
	  while x < S[0]:
	      y=0
	      while y < S[1]:
		  cluster_tuple=[0,0,0]#stores the x,y coordinate and the cluster number of each pixel
		  cluster_tuple[0]=x
		  cluster_tuple[1]=y
		  cluster_tuple[2]=CLUS(ARR[x,y],k)
		  CLUSTER_NO.append(cluster_tuple)
		  y=y+1
	      x=x+1
	  #finding new cluster means
	  NEW_MEAN(S,k,ARR)
	  flag=0
	  i=0
	  while i < k:
	      temp=MEAN_MOVE(i)
	      if temp!=0:
		  flag=1
	      i+=1
	  swap(k)
	  if flag==0:
	      break
	  NEW_CLUSTER_MEAN[:]=[]
	  CLUSTER_NO[:]=[]
   
      im2=Image.new("RGB",S,"#FFFFFF")
      ARR2=im2.load()
      color_arr=[]
      i=0
      while i < k:
	  c=[0,0,0]
	  c[0]=k
	  c[1]=0
	  c[2]=0
	  color_arr.append(c)
	  i+=1
      for i in range(len(CLUSTER_NO)):
	  v=(color_arr[CLUSTER_NO[i][2]][0],color_arr[CLUSTER_NO[i][2]][1],color_arr[CLUSTER_NO[i][2]][2])
	  im2.putpixel( ( CLUSTER_NO[i][0],CLUSTER_NO[i][1] ), v)
   
      im2.save(path+"/OUTPUT_"+str(k_nick)+"/"+f+e,"JPEG",quality=100)
      NEW_CLUSTER_MEAN[:]=[]
      CLUSTER_NO[:]=[]
      CLUSTER_MEAN[:]=[]
      SN+=1
