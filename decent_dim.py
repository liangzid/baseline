import numpy as np
import torch as t
'''
用来进行降维，利用均值和方差来代表一组数据，从而将整体的特征数据降维表出

liangzid,2018,8,20
'''

def dd(x,dAfter):
    d_after=dAfter*2
    length=len(x)
    if length%(d_after/2)!=0:
        print('*********************\nERROR!   \nERROR:降维数值有误\n**************************\n')
        return -1
    else:
        block_size=int(length/(d_after/2))
        dd1=np.zeros(int(d_after/2))
        dd2=np.zeros(int(d_after/2))
        
        for i in range(int(d_after/2)):
            xx=x[i*block_size:(i+1)*block_size]
            dd1[i]=get_mean(xx)
            dd2[i]=get_std(xx)
            
        result=np.concatenate((dd1,dd2))
        resultt=t.from_numpy(result).type(t.cuda.FloatTensor)    
        resultt=resultt.reshape(2,-1)
        
        return resultt

def get_mean(x):
    return x.mean()



#说是标准差，实际是方差
def get_std(x):
    mean=get_mean(x)
    std=((x-mean)**2).sum()
    return std



'''
A=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0])

B=dd(A,d_after=10)
print(B)

'''




