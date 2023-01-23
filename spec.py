
import psutil 

# Get CPU info 
cpu_info = psutil.cpu_times()

# Get Memory info 
memory_info = psutil.virtual_memory()

# Print out the CPU and Memory info 
print(cpu_info)
print(memory_info)