nums = [2, 7, 11, 15]
target = 9
l = len(nums)
for i in range(l):
    another = target - nums[i]
    if another in nums:  # 元素存在列表中
        j = nums.index(another)  #找索引
        if i==j:
            continue
        else:
            print([i,j])
