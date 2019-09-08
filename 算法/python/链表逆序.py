class Node(object): # 创建链表类，以题目为导向简化代码操作，只需要定义为包含当前节点的数值val和下一个节点指针next即可
    def __init__(self, data = None, next = None):
        self.val = data
        self.next = next
        
def fun1(head): # 实现方法一
    new_head = Node() # 用于存储逆序完的链表。初始化为null，目的是辅助第一个节点进行后续操作。
    while head!= None: # 当前节点不为空时
        temp_next = head.next # 先将当前节点原先指向的下一节点存储为临时变量temp_next
        head.next = new_head # 将当前节点连向下一节点的指针指向new_head
        new_head = head # 将当前节点保存为new_head
        head = temp_next # 将下一节点赋值给当前节点，继续循环操作
    return new_head
    
def fun2(head): # 实现方法二
    if head == None:
        return None
    L,M,R = None,None,head # 以3、2、1为例，R为3
    while R.next != None:  # 第一轮：R.next = 2。第二轮：R.next = 1
        L = M # 第一轮：L = none。第二轮：L = M = 3
        M = R # 第一轮：M = R = 3。第二轮：M = R = 2
        R = R.next # 第一轮：R = R.next = 2。第二轮：R = R.next = 1
        M.next = L # 第一轮：M.next原本R.next为2，现在设成L即none。第二轮：M.next = L = 3
    R.next = M 
    return R
    
if __name__ == '__main__':
    l1 = Node(3) 
    l1.next = Node(2)
    l1.next.next = Node(1)
    l1.next.next.next = Node(9) # 新建了一个顺序为3,2,1,9的链表
    l = fun1(l1)
    print (l.val, l.next.val, l.next.next.val, l.next.next.next.val) # 输出方法1结果
    l = fun2(l1)
    print (l.val, l.next.val, l.next.next.val, l.next.next.next.val) # 输出方法2结果
