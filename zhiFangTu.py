import matplotlib.pyplot as plt
numbers = []

with open("data.txt", "r") as file:
    # 逐行读取文件内容
    for line in file:
        numbers.extend([float(num) for num in line.strip().split(',')])


numbers = [min(num, 1) for num in numbers]







# 设置图表标题和轴标签
plt.title("SSIM distribution")
plt.xlabel("SSIM")
plt.ylabel("number")

bins, _, patches = plt.hist(numbers, bins=10, edgecolor='black')
for bin_value, patch in zip(bins, patches):
    height = patch.get_height()
    plt.text(patch.get_x() + patch.get_width() / 2, height, f"{round(bin_value/len(numbers),2)}", ha='center', va='bottom')

# 显示图表
plt.show()