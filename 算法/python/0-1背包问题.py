def solve(vlist,wlist,totalWeight,totalLength):
    resArr = [[0 for j in range(totalWeight + 1)] for i in range(totalLength + 1)] # 生成一个矩阵用于存放每次迭代的value值
    for i in range(1,totalLength+1): # 遍历到最多物品
        for j in range(1,totalWeight+1): # 遍历到最大载重
            if wlist[i] <= j: # 若当前物品的重量小于最大载重
                resArr[i][j] = max(resArr[i-1][j-wlist[i]]+vlist[i],resArr[i-1][j]) # 详情查背包问题公式
            else:
                resArr[i][j] = resArr[i-1][j]  
    return resArr

if __name__ == '__main__':
    v = [0,60,100,120] # 每种物品的价值，前补一位0
    w = [0,10,20,30] # 每种物品的重量，前补一位0
    weight = 50 # 最大载重
    n = 3 # 物品的个数
    result = solve(v,w,weight,n)
    print(result[-1][-1]) # 在最大载重下能达到的最大价值 
