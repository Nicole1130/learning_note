# 1.全遍历，数多的时候慢
num_list = [1, 2, 3, 4, 5]
print(num_list)
for item in num_list[:]:
    if item == 2:
        num_list.remove(item)
print(num_list)


# 2.倒叙，快，不会像正序一样出现序号溢出
num_list = [1, 2, 3, 4, 5]
print(num_list)
for i in range(len(num_list)-1, -1, -1):
    if num_list[i] == 2:
        num_list.pop(i)
print(num_list)
