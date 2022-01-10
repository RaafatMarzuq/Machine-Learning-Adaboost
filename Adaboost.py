# python 3.10.1

import itertools
import random
import numpy as np
import sys



class Point:
    def __init__(self, x=0, y=0, tag=0, w=0, used=False):
        self.x = x
        self.y = y
        self.tag = tag
        self.weight = w
        self.used = used

# line class 
class Rule:
       #init rule with two points : y=mx+n
    def __init__(self, point1, point2,up, error_rate=0.000001 ):
        self.point1 = point1
        self.point2 = point2
        self.up=up # up=true ->above the line is 1 , under the line is -1
        self.error_rate=error_rate
        if (point1.x   == point2.x ):
            self.parallelY = True 
            self.m = 0#it is the slope of the line
            self.n = 0
        else:
            self.parallelY = False
            self.m = (point1.y - point2.y) / (point1.x - point2.x)#it is the slope of the line
            self.n = point1.y - (self.m * point1.x)  #find the n int the equation y=mx+n



# Compute weighted error for h:
def compute_error(line, points):
    rate = 0.000000001
    for p in points:
        if (line.up==True): 
            if (p.y < line.m * p.x + line.n):
                if(p.tag!=-1):
                    rate+= p.weight
            else:
                if(p.tag!=1):
                    rate+= p.weight
        else: 
            if (p.y < line.m * p.x + line.n):
                if(p.tag!=1):
                    rate+= p.weight
            else:
                if(p.tag!=-1):
                    rate+= p.weight    
    return rate       



#Select classifier with min weighted error
def get_best_rule(points):
    min_error=np.inf
    rule=Rule(Point(),Point() , False)
    for p in itertools.combinations(points, 2):
        if (p[0].used==True and p[1].used==True):
            continue
        negative = Rule(p[0], p[1], True) #negative -> above the line is 1 
        rate=compute_error(negative, points)
        if(rate< min_error):
            rule=negative
            min_error=rate
            rule.error_rate=rate
        posetive= Rule(p[0],p[1],False)#posetive -> above the line is -1
        rate1=compute_error(posetive, points)
        if(rate1< min_error):
            rule=posetive
            min_error=rate1
            rule.error_rate=rate1
    rule.point1.used=True
    rule.point2.used=True

    return rule

#Set classifier weight 𝛼_𝑡 based on its error
def compute_alpha(error):
       
    return 0.5*np.log((1 - error) / error)


# h_i(x) , the predection of the point (p) with giving line 
def pred_point(line, p):
    if (line.up==True):
        if (p.y < line.m * p.x + line.n):
            return -1
        else:
            return 1
    else: 
        if (p.y < line.m * p.x + line.n):
            return 1
        else:
            return -1

# return true if the predection is right .
def h_i(line,p ):
    if (line.up==True):
        if (p.y < line.m * p.x + line.n):
            if(p.tag==-1):
                return 1
            else:
                return -1
        else:
            if(p.tag==1):
                return 1
            else:
                return -1
    else: 
        if (p.y < line.m * p.x + line.n):
            if(p.tag==1):
                return 1
            else:
                return -1
        else:
            if(p.tag==-1):
                return 1
            else:
                return -1

# 𝐻𝑘(𝑥) = 𝑠𝑖𝑔𝑛(∑ 𝛼𝑖 * ℎ𝑖(𝑥))
def H_K(k , point ,H, Alpha  ):
    sum =0
    for i in range(k) :
        sum += Alpha[i]*pred_point(H[i],point)
        
    if sum > 0:
        return 1
    else:
        return -1



def adaboost(points,test, rules):
    H=[] # H_K array 
    Alpha=[]
    # 1. Initialize point weights 𝐷1(𝑥𝑖)=1/𝑛.
    D_1 = 1 / len(points)
    for p in points:
        p.weight = D_1
        p.used=False
    test_acc =np.zeros(rules)
    train_acc =np.zeros(rules)
    for i in range(rules):
        total_sum=0
        # 3.    Compute weighted error for each h ∈ H:
        # 4.    Select classifier with min weighted error
        rule=get_best_rule(points)
        H.append(rule)

        # 5.    Set classifier weight 𝛼_𝑡 based on its error
        alpha=compute_alpha(rule.error_rate)
        Alpha.append(alpha)

        # Compute test Loss
        for p in test:
            if( H_K(i, p, H, Alpha) != p.tag):
                test_acc[i]+=1

         # Compute Train Loss
        for p in points:
            if( H_K(i, p, H, Alpha) != p.tag):
                train_acc[i]+=1

        #6.    Update point weights
        for p in (points):
            if(h_i(rule,p)==1): 
                p.weight *= (np.math.e ** (-1 * alpha))

            else :
                p.weight *= (np.math.e ** alpha)
            
            total_sum += p.weight
       

    #normalizing
        for p in (points):
             p.weight /=total_sum
  


    test_acc /= len(test)
    train_acc /= len(points)

    return test_acc, train_acc


def main(all_points , rules , round):
    train_acc = np.zeros(8)
    test_acc = np.zeros(8)
    points=all_points.copy()
    for j in range(100):
        new_points = points.copy()
        random.shuffle(new_points)
        Train = new_points[0:75]
        Test = new_points[75:150]
        temp2, temp1 = adaboost(Train,Test,8)
        test_acc += temp2
        train_acc += temp1
   
    for i in range(rules):
        print("Train: {} Rules \nError is {:.2f}% ".format(i+1, train_acc[i]))
        print("Test:  {} Rules \nError is {:.2f}% ".format(i+1, test_acc[i]))
        print("")

if __name__ == '__main__':
    all_points = []
    f = open("rectangle.txt", "r")
    for line_data in f:
        single_point = line_data.split()
        single_point = Point(float(single_point[0]), float(single_point[1]), float(single_point[2]))
        all_points.append(single_point)
    main(all_points, 8, 100)
