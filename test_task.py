import time
import glfw
import numpy as np
import argparse
import random
import metaworld
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_reach_push_pick_place import SawyerReachPushPickPlaceEnv
from metaworld import Benchmark, _env_dict, _make_tasks, _ML_OVERRIDE

def sample_sawyer_reach_push_pick_place():
    ml1 = metaworld.ML1('pick-place-v1')
    env = SawyerReachPushPickPlaceEnv()
    task = random.choice(ml1.train_tasks)
    env.set_task(task)  # Set task
    env.reset()
    for i in range(100):
        if i % 100 == 0:
            env.reset()
        env.step(env.action_space.sample())
        env.render()
    glfw.destroy_window(env.viewer.window)

def test_door1():
    ml10 = metaworld.ML10()
    env = ml10.train_classes['door-open-v1']()
    task = random.choice([task for task in ml10.train_tasks if task.env_name == 'door-open-v1'])
    env.set_task(task)

    done = False
    # obs = env.reset()
    obs = env.reset_with_pos([-0.03265005, 0.51488463, 0.2368774], [0.12760278, 0.72007364, 0.15])
    print(obs)


    while not done:
        obs, reward, done, info = env.step(
            env.action_space.sample())  # Step the environoment with the sampled random action
        env.render(mode='human')
        time.sleep(.05)

def test_reset():
    ml1 = metaworld.ML1('push-v1')
    env = ml1.train_classes['push-v1']()
    task = random.choice(ml1.train_tasks)
    env.set_task(task)

    done = False
    obs = env.reset()
    print(obs)

    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())  # Step the environoment with the sampled random action
        env.render(mode='human')
        time.sleep(.05)

def test_reach():
    ml1 = metaworld.ML1('push-v1')
    env = ml1.train_classes['push-v1']()
    task = random.choice(ml1.train_tasks)
    env.set_task(task)  # Set task

    done = False
    env.reset()
    step = 0 # max_path_length = 300
    print(env)
    file_action = open('/home/yxt/Research/RL/learning_to_be_taught/action_saved/reach1.txt', 'r')
    action = []
    count = 0
    for line in file_action.readlines():
        line = line.strip().split()
        action.append([float(line[0]), float(line[1]), float(line[2]), float(line[3])])
        count += 1
    # goal = np.array([action[0][0], action[0][1], action[0][2]])
    # print(goal)
    # action = action[1:-1]
    # env._set_pos_site('goal_reach', goal)
    # print(env._get_pos_goal())
    raction = np.negative(action[::-1])
    raction[:,-1] *= -1.0

    # print(action[1])
    # print(raction[1])
    # while True:
    #     pass
    i = 0
    a = action[0]
    reverse_flag = False
    while not done:
        if i >= 2*count:
            # while True:
            #     pass
            i = 0
            a = action[0]
            reverse_flag = False
        if i < count:
            pass
        else:
            reverse_flag = True
        # a = env.action_space.sample()  # Sample an action

        if not reverse_flag:
            a = action[i]
            print(f'step {step} : {env.get_endeff_pos()}')
            print(f'step {step} : {a}')
        else:
            a = raction[i-count]
            print('reverse sample:')
            print(f'step {2*count-step-1} : {env.get_endeff_pos()}')
            print(f'step {2*count-step-1} : {a}')

        obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
        env.render(mode='human')
        time.sleep(.05)
        # joint = np.array(
        #     ['kuka_joint_1', 'kuka_joint_2', 'kuka_joint_3',
        #      'kuka_joint_4', 'kuka_joint_5', 'kuka_joint_6',
        #      'kuka_joint_7']
        # )

        # for i in joint:
        #     print(f'{i}: {env.get_joint_qpos(i)}')
        print("\n")
        step = step + 1
        i+=1
    print('Done.')

def test_drawer():
    ml10 = metaworld.ML10()
    test_task = 'door-open-v1'
    for name, env_cls in ml10.train_classes.items():
        if name == test_task:
            env = env_cls()
            task = random.choice([task for task in ml10.train_tasks if task.env_name == test_task])
            env.set_task(task)

    done = False
    env.reset()
    step = 0 # max_path_length = 300
    print(env)
    file_action = open('/home/yxt/Research/RL/learning_to_be_taught/action_saved/ddoor0.txt','r')
    action = []
    count = 0
    for line in file_action.readlines():
        line = line.strip().split()
        action.append([float(line[0]), float(line[1]), float(line[2]), float(line[3])])
        count += 1
    # goal = np.array([action[0][0], action[0][1], action[0][2]])
    # print(goal)
    # action = action[1:-1]
    # env._set_pos_site('goal_reach', goal)
    # print(env._get_pos_goal())
    raction = np.negative(action[::-1])
    raction[:,-1] *= -1.0

    # print(action[1])
    # print(raction[1])
    # while True:
    #     pass
    i = 0
    a = action[0]
    reverse_flag = False
    while not done:
        if i >= 2*count:
            # while True:
            #     pass
            i = 0
            a = action[0]
            reverse_flag = False
        if i < count:
            pass
        else:
            reverse_flag = True
        # a = env.action_space.sample()  # Sample an action

        if not reverse_flag:
            a = action[i]
            print(f'step {step} : {env.get_endeff_pos()}')
            print(f'step {step} : {a}')
        else:
            a = raction[i-count]
            print('reverse sample:')
            print(f'step {2*count-step-1} : {env.get_endeff_pos()}')
            print(f'step {2*count-step-1} : {a}')

        obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
        env.render(mode='human')
        time.sleep(.05)

        print("\n")
        step = step + 1
        i+=1
    print('Done.')

def test_single_task(name, num):
    # read data from files
    file_action = open('/home/yxt/Research/RL/data/learning_to_be_taught/action_saved/'+name+'-'+str(num)+'.txt', 'r')
    action = []
    hand_init_pos = [-0.03265, 0.51488, 0.23687]
    move_left_right = np.array([[-1, 0, 0, 0],[1, 0 ,0, 0]])
    move_back_front = np.array([[0, -1, 0, 0],[0, 1, 0, 0]])
    move_up_down = np.array([[0, 0, 1, 0],[0, 0, -1, 0]])

    for line in file_action.readlines():
        line = line.strip().split()
        action.append([float(line[0]), float(line[1]), float(line[2]), float(line[3])])
    action = action[3:] #block abnormal data [0:4]
    raction = action
    raction = np.negative(raction[::-1])
    raction[:, 2:4] *= -1.0
    # raction[:, -1] *= -1.0
    print(action[0])
    print(raction[-1])

    file_obs = open('/home/yxt/Research/RL/data/learning_to_be_taught/obs_saved/'+name+'-'+str(num)+'.txt', 'r')
    obs = []
    for line in file_obs.readlines():
        line = line.strip().split()
        obs.append([float(line[3]), float(line[4]), float(line[5])])

    if name == 'door-open-v1':
        obs_pos1 = obs[0] + np.array([-0.1, 0.15, -0.05]) # door-open
    elif name == 'drawer-close-v1':
        obs_pos1 = obs[0] + np.array([0, 0.4, -0.05]) # drawer-close
    else:
        pass
    # print(obs[0])
    # print("----------------")

    ml10 = metaworld.ML10_Train_Single(name, obs_pos1)
    env = ml10.train_classes[name]()
    task = ml10.train_tasks[0]
    env.set_task(task)  # Set task
    obs = env.reset()
    print(f'reset 1 :{obs}')
    done = False
    print(obs)

    i = 0
    max_length = False
    a = None
    count = np.shape(action)[0]
    # while not done:
    #     if i >= np.shape(action)[0] and max_length:
    #         env.reset()
    #         i = 0
    #     a = action[i]
    #     obs, reward, done, info = env.step(a)  # Step the environment with the sampled random action
    #     print(f'step {i}: action{a}, obs{obs}')
    #     if info['success']:
    #         max_length = True
    #     env.render(mode='human')
    #     time.sleep(.05)
    #     i = i+1

    reach_act = np.zeros([1,4])
    move_up_down[0][-1] = action[-1][-1]
    move_up_down[1][-1] = action[-1][-1]
    move_back_front[0][-1] = action[-1][-1]
    move_back_front[1][-1] = action[-1][-1]
    move_left_right[0][-1] = action[-1][-1]
    move_left_right[1][-1] = action[-1][-1]
    reach_act[0] = move_up
    curr_hand_pos = np.zeros([1,3])
    switch = [0,0,0]

    flag = False
    while not done:
        if i < count:
            a = action[i]
        else:
            a = raction[i-count]
        obs, reward, done, info = env.step(a)  # Step the environment with the sampled random action
        print(f'step {i}: action{a}, obs{obs}')
        if info['success']:
            max_length = True
        env.render(mode='human')
        time.sleep(.02)



        i = i+1
        if i == count:
            if obs[0] < hand_init_pos[0]:
                switch[0] = 1
            if obs[1] < hand_init_pos[1]:
                switch[1] = 1
            if obs[2] > hand_init_pos[2]:
                switch[2] = 1
            env.reset_curr_path_length()
            cnt = 0
            curr_hand_pos = obs[:3]
            while True:
                if obs[2] < hand_init_pos[-1]:
                    reach_act = np.append(reach_act, np.array(move_up_down[0]).reshape([1, 4]), axis=0)
                    obs, reward, done, info = env.step(reach_act[-1])
                elif obs[1] > hand_init_pos[1]:
                    reach_act = np.append(reach_act, np.array(move_back_front[0]).reshape([1, 4]), axis=0)
                    obs, reward, done, info = env.step(reach_act[-1])
                elif switch[0]==1:
                    reach_act = np.append(reach_act, np.array(move_left_right[1]).reshape([1, 4]), axis=0)
                    obs, reward, done, info = env.step(reach_act[-1])
                elif switch[0]==0:
                    reach_act = np.append(reach_act, np.array(move_left_right[0]).reshape([1, 4]), axis=0)
                    obs, reward, done, info = env.step(reach_act[-1])
                else:
                    rreach_act = np.negative(reach_act)
                    rreach_act[:,-1] *= -1.0
                    rcount = np.shape(rreach_act)[0]
                    while rcount>1:
                        obs, reward, done, info = env.step(rreach_act[rcount-1])
                        rcount -= 1
                        print("xxx")
                        flag = True
                        env.render(mode='human')
                        time.sleep(.03)
                if flag:
                    break
                print(f"num:{cnt}, hand:{obs[:3]}, before:{curr_hand_pos}")
                cnt += 1
                env.render(mode='human')
                time.sleep(.03)

        if i-count >= np.shape(raction)[0]:
            break
    print("done.")


def test_single():
    obs_pos = np.array([0.0463498, 0.94044094, 0.1])
    ml10 = metaworld.ML10_Train_Single('door-open-v1', obs_pos)
    env = ml10.train_classes['door-open-v1']()
    task = ml10.train_tasks[0]
    env.set_task(task)  # Set task

    obs = env.reset()
    print(obs)
    done = False

    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())  # Step the environoment with the sampled random action
        env.render(mode='human')
        time.sleep(.05)


if __name__ == '__main__':
    #sample_sawyer_reach_push_pick_place()
    # test_reach()
    name = 'door-open-v1'
    # name = 'drawer-close-v1'
    num = 0
    test_single_task(name, num)