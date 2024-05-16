import subprocess

command = "python widar3_keras.py 1"  # 请将 "your_command_here" 替换为您要执行的实际命令

results = []
for _ in range(10):
    result = subprocess.check_output(command, shell=True)
    result = result.decode().strip()  # 如果结果是字节字符串，请解码并去除首尾空格

    # 在这里添加适当的字符串处理方法来提取最后一行的数据
    # 以下是一个示例，假设结果是多行字符串，每行以换行符分隔
    lines = result.split("\n")  # 使用换行符分割字符串
    last_line = lines[-1]  # 获取最后一行

    extracted_result = float(last_line)  # 将最后一行转换为浮点数

    print(extracted_result)
    results.append(extracted_result)  # 将提取的结果添加到结果列表中

average = sum(results) / len(results)  # 计算结果的平均值

print("平均值：", average)

# 0.9623333333333333
# 0.933968253968254
# 0.9257575757575758
# 0.8962318840579708
# 0.8991666666666669
# 0.8957333333333333

# 0.21333333333333335
# 0.2733333333333333
# 0.29523809523809524
# 0.30833333333333335
# 0.28888888888888886
# 0.30666666666666664

