# 1.冒泡法排序，时间复杂度是O(n2)
def bubble_sort(li): # 输入li列表
    for j in range(len(li)-1):
        for i in range(1, len(li)):
            if li[i] > li[i-1]:
                li[i], li[i-1] = li[i-1], li[i]
    return li

# 2.插入排序，时间复杂度是O(n2)
def insert_sort(li):
    for i in range(1,len(li)):
        tmp = li[i]
        j = i - 1
        while j >= 0 and tmp < li[j]:　　　　# 找到一个合适的位置插进去
            li[j+1] = li[j]
            j -= 1
        li[j+1] = tmp
    return li

# 3.选择排序，时间复杂度是O(n2)
def select_sort(li):
    for i in range(len(li)-1):
        min_loc = i         # 假设当前最小的值的索引就是i
        for j in range(i+1,len(li)):
            if li[j] < li[min_loc]:
                min_loc = j
        if min_loc != i:   # min_loc 值如果发生过交换，表示最小的值的下标不是i,而是min_loc
            li[i],li[min_loc] = li[min_loc],li[i]
 
    return li

# 4.快速排序，时间复杂度是O(nlogn)
def partition(arr,low,high): # 
    i = low # 起始位置索引
    pivot = arr[high] # 以最高索引位置的元素作为基准值       
    for j in range(low , high):  # 遍历区间内元素
        if arr[j] <= pivot:  # 当前元素小于或等于基准值 
            arr[i],arr[j] = arr[j],arr[i]   # 将这个元素与起始位置的元素进行交换
            i = i+1 # 后移起始位置
    arr[i],arr[high] = arr[high],arr[i]  # 遍历完毕后将最高位置元素（基准值）存到起始位置
    return ( i ) # 返回这次基准值调整到的位置

def quickSort(arr,low,high): # arr[] --> 排序数组,low  --> 起始索引,high  --> 结束索引
    if low < high:  
        pi = partition(arr,low,high) # pi为基准值调整到的位置索引
        quickSort(arr, low, pi-1)  # 将区间2分再进行快排
        quickSort(arr, pi+1, high) 
# 调用
arr = [10, 7, 8, 9, 1, 5] 
n = len(arr) 
quickSort(arr,0,n-1) 


# 4.归并排序，时间复杂度是O(nlogn)，O(n)的空间复杂度
def mergeSort(alist):
    if len(alist)>1:
        mid = len(alist)//2
        lefthalf = alist[:mid]
        righthalf = alist[mid:] # 分为左右两部分
        mergeSort(lefthalf) # 递归分别求两部分的归并排序
        mergeSort(righthalf)
        i=0 # 用于记录lefthalf部分索引位置
        j=0 # 用于记录righthalf部分索引位置
        k=0 # 用于记录暂存列表alist中索引位置
        while i<len(lefthalf) and j<len(righthalf): # 当左右两边均没有遍历完时
            if lefthalf[i]<righthalf[j]: # 左边和右边的元素分别比较，小的存入alist中
                alist[k]=lefthalf[i]
                i=i+1
            else:
                alist[k]=righthalf[j]
                j=j+1
            k=k+1

        while i<len(lefthalf): # 当右侧right部分已经遍历结束，而左侧left部分仍有剩余时将剩余部分直接存入alist
            alist[k]=lefthalf[i]
            i=i+1
            k=k+1

        while j<len(righthalf): # 同理左侧left部分已经遍历结束，而右侧right部分仍有剩余
            alist[k]=righthalf[j]
            j=j+1
            k=k+1
# 调用
alist = [54,26,93,17,77,31,44,55,20]
mergeSort(alist)
print(alist)

# 5.堆排序,时间复杂度O(nlogn)
def sift(li, left, right):  # left和right 表示了元素的范围，是根节点到右节点的范围，然后比较根和两个孩子的大小，把大的放到堆顶
                                    # 和两个孩子的大小没关系，因为我们只需要拿堆顶的元素就行了
    # 构造堆
    i = left        # 当作根节点
    j = 2 * i + 1   # 上面提到过的父节点与左子树根节点的编号下标的关系
    tmp = li[left]
    while j <= right:
        if j+1 <= right and li[j] < li[j+1]:    # 找到两个孩子中比较大的那个
            j = j + 1
        if tmp < li[j]:     # 如果孩子中比较大的那个比根节点大，就交换
            li[i] = li[j]
            i = j           # 把交换了的那个节点当作根节点，循环上面的操作
            j = 2 * i + 1
        else:           
            break
    li[i] = tmp             # 如果上面发生交换，现在的i就是最后一层符合条件（不用换）的根节点，
 
def heap_sort(li):
    n = len(li)
    for i in range(n//2-1, -1, -1):  # 建立堆        n//2-1 是为了拿到最后一个子树的根节点的编号，然后往前走，最后走到根节点0//2 -1 = -1
        sift(li, i, n-1)                # 固定的把最后一个值的位置当作right，因为right只是为了判断递归不要超出当前树，所以最后一个值可以满足
                                                    # 如果每遍历一个树，就找到它的右孩子，太麻烦了
    for i in range(n-1, -1, -1):    # 挨个出数
        li[0], li[i] = li[i],li[0]      # 把堆顶与最后一个数交换，为了节省空间，否则还可以新建一个列表，把堆顶（最大数）放到新列表中
        sift(li, 0, i-1)            # 此时的列表，应该排除最后一个已经排好序的，放置最大值的位置，所以i-1

# 用heapq模块来实现堆排序
import heapq
def heapq_sort(li):
    h = []
    for value in li:
        heapq.heappush(h,value)
    return [heapq.heappop(h) for i in range(len(h))] 
heapq.nlargest(10,li) # 取top10
 
# 直接调用python的sort函数进行排序    
L = input("Enter your list: ")
L = L.split(",")  #分割
Lstr = list(L) #列表化
Lint = [int(Lstr[i]) for i in range(len(Lstr))] #逐个转类型
print (Lint)
#print([Lint[i] for i in range(len(Lint))]) #整体输出
#for i in range(len(Lint)): #单个输出
#    print(Lint[i])
Lint.sort(reverse = True) #降序
print (Lint[:2]) #输出前两位，左闭右开
