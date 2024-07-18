import numpy as np

class ResourceState:
    def __init__(self, cpu, gpu, memory, disk, network):
        self.cpu = cpu
        self.gpu = gpu
        self.memory = memory
        self.disk = disk
        self.network = network

    def get_state(self):
        return np.array([self.cpu, self.gpu, self.memory, self.disk, self.network])


class Task:
    def __init__(self, task_type, scale, memory, disk, state, dependencies):
        self.task_type = task_type
        self.scale = scale
        self.memory = memory
        self.disk = disk
        self.state = state
        self.dependencies = dependencies

    def get_state(self):
        return np.array([self.task_type, self.scale, self.memory, self.disk, self.state, len(self.dependencies)])


class Scheduler:
    def calculate_transmission_time(subtask1, subtask2, bandwidth):
        """
        计算两个子任务之间的数据传输时间
        """
        if subtask1.server == subtask2.server:
            # 两个子任务在同一服务器上，传输时间可以忽略
            return 0
        else:
            min_bandwidth = min(subtask1.server.bandwidth, subtask2.server.bandwidth)
            data_size = subtask2.size  # 假设数据传输大小等于子任务2的大小
            return data_size / min_bandwidth

    def calculate_start_time(self, subtask, processor, bandwidth):
        # 考虑数据传输时间、前驱子任务完成时间和处理器空闲时间
        if subtask.predecessor_id is not None:
            predecessor = self.get_subtask_by_id(subtask.predecessor_id)
            transmission_time = self.calculate_transmission_time(subtask, predecessor, bandwidth)
        else:
            transmission_time = 0

        start_time = max(
            processor.idle_time,
            self.get_subtask_by_id(subtask.predecessor_id).completion_time if subtask.predecessor_id else 0
        ) + transmission_time

        return start_time

    def idle_time(processor):

        current_time = max(processor.start_time)

        for start, duration in zip(processor.start_time, processor.duration):
            if current_time < start + duration:
                current_time = start + duration

        return current_time

    def calculate_task_completion_time(task, processors, bandwidth):

        start_times = {}
        completion_times = {}

        for subtask in task.subtasks:
            start_times[subtask.id] = 0
            completion_times[subtask.id] = 0

        for subtask in task.subtasks:

            exec_time = subtask.size / processors[subtask.processor].speed

            start_times[subtask.id] = max(
                task.idle_time(processors[subtask.processor]),
                completion_times.get(subtask.predecessor.id, 0)
            )

            trans_time = 0
            if subtask.predecessor:
                trans_time = task.calculate_transmission_time(subtask, subtask.predecessor, bandwidth)

            start_times[subtask.id] += trans_time

            completion_times[subtask.id] = start_times[subtask.id] + exec_time

        return max(completion_times.values())

    def calculate_power_consumption(task, processors):

        dynamic_power = 0
        static_power = 0

        for subtask in task.subtasks:
            dynamic_power += processors[subtask.processor].power * subtask.size

        static_power = (3 / 7) * dynamic_power

        return dynamic_power + static_power

    def calculate_resource_utilization(server):

        cpu_utilization = server.cpu_usage / server.cpu_capacity
        gpu_utilization = server.gpu_usage / server.gpu_capacity
        memory_utilization = server.memory_usage / server.memory_capacity
        network_utilization = server.network_usage / server.network_capacity
        disk_utilization = server.disk_usage / server.disk_capacity

        # 返回资源使用率字典
        return {
            'cpu': cpu_utilization,
            'gpu': gpu_utilization,
            'memory': memory_utilization,
            'network': network_utilization,
            'disk': disk_utilization
        }

    def calculate_load_balance(task, processors, bandwidth):

        server_time = {server.id: 0 for server in processors}

        for subtask in task.subtasks:
            server_time[processors[subtask.processor].server.id] += subtask.size / processors[subtask.processor].speed

        load_time = min(server_time.values()) / max(server_time.values())

        load_resource = 0
        for server in processors:

            cpu_util = server.cpu.utilization
            gpu_util = server.gpu.utilization
            mem_util = server.memory.utilization
            net_util = server.network.utilization
            disk_util = server.disk.utilization

            v_cpu = cpu_util / server.cpu.capacity
            v_gpu = gpu_util / server.gpu.capacity
            v_mem = mem_util / server.memory.capacity
            v_net = net_util / server.network.capacity
            v_disk = disk_util / server.disk.capacity

            cpu_ratio = cpu_util / v_cpu
            gpu_ratio = gpu_util / v_gpu
            mem_ratio = mem_util / v_mem
            net_ratio = net_util / v_net
            disk_ratio = disk_util / v_disk

            load_resource += (task.cpu_utilization(cpu_ratio) * server.weight_cpu +
                              task.gpu_utilization(gpu_ratio) * server.weight_gpu +
                              task.memory_utilization(mem_ratio) * server.weight_mem +
                              task.network_utilization(net_ratio) * server.weight_net +
                              task.disk_utilization(disk_ratio) * server.weight_disk)

        load_resource /= len(processors)

        alpha = 0.5
        sys_load = alpha * load_time + (1 - alpha) * load_resource

        return sys_load

    def objective_function(task, processors, bandwidth):

        max_completion_time = task.calculate_task_completion_time(task, processors, bandwidth)

        total_power_consumption = task.calculate_power_consumption(task, processors)

        load_balance = task.calculate_load_balance(task, processors, bandwidth)

        return max_completion_time, total_power_consumption, load_balance
