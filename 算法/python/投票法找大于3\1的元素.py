nums = [1,2,3,2,2,1,3,1,2,5,2,4,8,0,7,5,6]

num1,count1 = nums[0],0
num2,count2 = nums[1],0
for x in nums: # 投票法找出主要元素的候选值
    if x == num1:
        count1 += 1
    elif x == num2:
        count2 += 1
    elif count1 == 0:
        num1,count1 = None,0
    elif count2 == 0:
        num2,count2 = None,0
    else:
        count1 -= 1
        count2 -= 1
        
count1,count2 = 0,0        
for x in nums: # 验证候选值是否大于1/3
    if x == num1:
        count1 += 1
    if x == num2:
        count2 += 1
res = []
if count1 > len(nums)//3:
    res.append(num1)
if count2 > len(nums)//3:
    res.append(num2)    
print(res) # 输出大于1/3的元素
