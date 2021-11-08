import subprocess
import argparse
import time

# 参数
parser = argparse.ArgumentParser()
parser.add_argument('--map_type', type=str)
parser.add_argument('--map_size', type=str)
parser.add_argument('--agent_type', type=str)
parser.add_argument('--gpu_index', type=int)
parser.add_argument('--task_para_num', type=int)
FLAGS, unparsed = parser.parse_known_args()

# 由于显存的大小区别，分类map
same_type_map = {
    'small': ["2m_vs_1z" ,"2s_vs_1sc", "3m", "3s_vs_3z", "3s_vs_4z", "3s_vs_5z", "5m_vs_6m"],
    'median': ["6h_vs_8z" ,"8m" ,"8m_vs_9m", "10m_vs_11m"],
    'large': ["25m" ,"27m_vs_30m", "corridor", "so_many_baneling", "2c_vs_64zg"]
}

diff_type_map = {
    'small': ["2s3z", "3s_vs_5z"],
    'median': ["3s5z" ,"3s5z_vs_3s6z" , "MMM" ,"MMM2"],
    'large': ["bane_vs_bane"]
}

map_scale = {
    'same_type': same_type_map,
    'diff_type': diff_type_map
}

task_list = []
# 还有任务可以跑
while len(map_scale[FLAGS.map_type][FLAGS.map_size]) != 0:
    map_name = map_scale[FLAGS.map_type][FLAGS.map_size].pop(0)
    task = subprocess.Popen(["CUDA_VISIBLE_DEVICES='{}' python src/main.py --config=qmix --env-config=sc2 with env_args.map_name={} agent='{}' asn_hidden_size=32 t_max=100".format(FLAGS.gpu_index, map_name, FLAGS.agent_type)], shell=True)
    task_list.append(task)
    time.sleep(3)

    if len(task_list) == FLAGS.task_para_num or (len(task_list) + len(map_scale[FLAGS.map_type][FLAGS.map_size]) < FLAGS.task_para_num):
        jump = False
        while not jump:
            for task in task_list:
                if task.poll() is not None:
                    task_list.remove(task)
                    jump = True
                    break
                else:
                    time.sleep(10)

# 收尾下剩下的
while len(task_list) != 0:
    for task in task_list:
        if task.poll() is not None:
            task_list.remove(task)
        else:
            time.sleep(10)
