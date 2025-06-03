import re

# 读取result.txt文件
with open(r'c:\Users\Perception\Desktop\Obstacle_Avoidance\Obstacle_Avoidance_2024\result.txt', 'r') as f:
    lines = f.readlines()

# 使用正则表达式提取inference时间
inference_times = []
pattern = r'(\d+\.\d+)ms inference'

for line in lines:
    match = re.search(pattern, line)
    if match:
        inference_time = float(match.group(1))
        inference_times.append(inference_time)

# 计算平均时间
if inference_times:
    avg_time = sum(inference_times) / len(inference_times)
    print(f"Total number of samples: {len(inference_times)}")
    print(f"Average inference time: {avg_time:.2f}ms")
else:
    print("No inference times found in the file.")
