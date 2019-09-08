# 二分法查找，时间复杂度O(logn)
def binary_serach(li,val):  # 输入li列表，val查找的值
    low = 0 # 左端点 
    high = len(li)-1 # 右端点
    while low <= high:
        mid = (low+high)//2 # //表示整数除法，/表示浮点型。二分法中点
        if li[mid] == val: # 中点值正好等于检索值
            return mid
        elif li[mid] > val: # 中点值大于检索值
            high = mid-1  # 右端点设置到中点附近
        else:
            low = mid+1
    else:
        return None
        
binary_serach([1,2,3,4,5],5) #调用函数
