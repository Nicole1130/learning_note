matrix = [[1,2,3],[4,5,6],[7,8,9]]
n = len(matrix)
for i in range(n):
    new_row = [row[i] for row in matrix[:n][::-1]]  # 对每一行的列进行翻转
    matrix.append(new_row)
del matrix[:n]   #删除matrix中原有的数据[:n]行
