s = 'ABCYDYEATTMBQECPD' # 输入全部序列
target = 'ABCDE' # 目标序列
cut = None
l = len(s)
min_l = l # 最短序列长度。初始化为全部序列的长度。
for i in range(l) : # 循环遍历，用于序列移动后的重排检测
    x = []
    for j in target : # 遍历目标序列中的每一个元素
        add = s.find(j) # 在全部序列中找到当前元素的位置序号
        x.append(add) # 把位置序号存入x
    x.sort() # 将所有位置序号排序
    if min_l > x[-1] : # 若序号最后一位比最小标志小
        min_l = x[-1] # 将最小标志设为最后一位。只考虑最后一位的原因是
        cut = s [x[0]:x[-1]+1]
    s = s[1:] + s[0] # 将整体序列右移一位
print("最短序列长度:",min_l)
print("最短序列:",cut)
