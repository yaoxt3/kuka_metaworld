import numpy as np
from gym.spaces import Box
import jpype
from jpype.types import *
from metaworld.envs.asset_path_utils import full_v1_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set

import sys
sys.path.append("/home/yxt/Research/RL/RealKukaEnv/")
from interface import KukaInterface

class SawyerReachPushPickPlaceKukaS2REnv(SawyerXYZEnv):

    def __init__(self):
        liftThresh = 0.04
        goal_low = (0.65, -0.1, 0.05)
        goal_high = (0.75, 0.1, 0.2)
        hand_low = (0.45, -0.45, 0.05)
        hand_high = (0.85, 0.45, 0.3)
        obj_low = (0.45, -0.1, 0.02)
        obj_high = (0.55, 0.1, 0.02)

        #self.kuka_initial_pos = [0.326, 0.906, 0.0, -1.376, 0.0, 0.86, 0.336, 0.0, 0.0] # pos: [0.58521036 0.19782366 0.04969333]
        self.kuka_initial_pos = [0.0, 0.3348, 0.0, -1.885, 0.0, 0.9221, 0.0, 0.0, 0.0] # pos: [0.45, 0.0, 0.2]
        self.kuka_env = None
        self.sim2real = False
        self.fingerHeight = -0.0535

        # self.kuka_goal = np.array([0.705, -0.063, 0.117]) # point 1
        # self.kuka_goal = np.array([0.705, -0.063, 0.15]) # point 2
        self.kuka_goal = np.array([0.665, 0.01, 0.11]) # point 3
        self.task_types = ['pick_place', 'reach', 'push']

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.task_type = None
        self.init_config = {
            'obj_init_angle': .3,
            'obj_init_pos': np.array([0.45, 0, 0.02]),
            'hand_init_pos': np.array([0.45, 0, 0.2]),
            #'hand_init_pos': np.array([0.6, 0.197, 0.05]),
        }

        self.obj_init_angle = self.init_config['obj_init_angle']
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.liftThresh = liftThresh

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.num_resets = 0



    ########### kuka sim2real interface ###########
    def kuka_set_env(self):
        self.kuka_env = KukaInterface(self.kuka_initial_pos)

    def kuka_reset_env(self):
        #self.init_fingerCOM = self.kuka_env.getCurrentGripperPosition()
        self.init_fingerCOM = self.kuka_env.getCurrentFrame() + np.array([0.0, 0.0, self.fingerHeight])
        self.maxReachDist = np.linalg.norm(self.init_fingerCOM - np.array(self.kuka_get_pos_goal()))
        self.maxPushDist = np.linalg.norm(self.kuka_get_pos_obj()[:2] - np.array(self.kuka_get_pos_goal())[:2])
        self.maxPlacingDist = np.linalg.norm(np.array([self.kuka_get_pos_obj()[0], self.kuka_get_pos_obj()[1], self.heightTarget]) - np.array(self.kuka_get_pos_goal())) + self.heightTarget
        self.target_rewards = [1000*self.maxPlacingDist + 1000*2, 1000*self.maxReachDist + 1000*2, 1000*self.maxPushDist + 1000*2]

        if self.task_type == 'reach':
            idx = 1
        elif self.task_type == 'push':
            idx = 2
        elif self.task_type == 'pick_place':
            idx = 0
        else:
            raise NotImplementedError

        self.target_reward = self.target_rewards[idx]
        self.num_resets += 1

        return self.kuka_get_obs()


    def kuka_set_initial_pos(self, initial_pos):
        self.kuka_initial_pos = initial_pos

    def set_sim2real(self, bool):
        self.sim2real = bool

    def kuka_get_pos_obj(self):
        if self.task_type == 'pick_place':
            pass
        elif self.task_type == 'reach':
            return np.array([0.54497565, -0.05953481, 0.01499632])
        elif self.task_type == 'push':
            pass
        else:
            raise NotImplementedError

    def kuka_set_pos_goal(self, goal):
        self.kuka_goal = goal

    def kuka_get_pos_goal(self):
        return self.kuka_goal

    def kuka_get_obs(self):
        pos_hand = self.kuka_env.getCurrentFrame()
        pos_obj_padded = np.zeros(6)
        pos_obj = self.kuka_get_pos_obj()
        pos_obj_padded[:len(pos_obj)] = pos_obj
        pos_goal = self.kuka_get_pos_goal()

        if self._partially_observable:
            pos_goal = np.zeros_like(pos_goal)

        return np.hstack((pos_hand, pos_obj_padded, pos_goal))


    def kuka_step(self, action):
        print("kuka step")
        print("kuka hand pos", self.kuka_env.getCurrentFrame())
        print("kuka goal pos", self.kuka_get_pos_goal())
        print("kuka obj pos", self.kuka_get_obs())
        action = action.copy()
        pos_ctrl = action[:3] * 0.01
        print(f'pos_ctrl: {pos_ctrl}')
        move = np.array([float(pos_ctrl[0]), float(pos_ctrl[1]), float(pos_ctrl[2])])
        self.kuka_env.setAction(move)
        self.kuka_env.step()
        print("kuka step end")

        return self.kuka_get_obs()

    def kuka_compute_reward(self, actions, obs):

        objPos = obs[3:6]

        #fingerCOM  =  self.kuka_env.getCurrentGripperPosition()
        fingerCOM  =  self.kuka_env.getCurrentFrame() + np.array([0.0, 0.0, self.fingerHeight])
        print("kuka fingerCOM", fingerCOM)

        heightTarget = self.heightTarget
        goal = self.kuka_get_pos_goal()

        def compute_reward_reach(actions, obs):
            del actions
            del obs

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            reachDist = np.linalg.norm(fingerCOM - goal)
            print("reachDist", reachDist)
            print("maxReachDist", self.maxReachDist)

            reachRew = c1*(self.maxReachDist - reachDist) + c1*(np.exp(-(reachDist**2)/c2) + np.exp(-(reachDist**2)/c3))
            print("reachRew", reachRew)

            reachRew = max(reachRew, 0)
            reward = reachRew

            print("compute reward reach", reward)
            return [reward, reachRew, reachDist, None, None, None, None, None]

        def compute_reward_push(actions, obs):
            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            del actions
            del obs

            assert np.all(goal == self._get_site_pos('goal_push'))
            reachDist = np.linalg.norm(fingerCOM - objPos)
            pushDist = np.linalg.norm(objPos[:2] - goal[:2])
            reachRew = -reachDist
            if reachDist < 0.05:
                pushRew = 1000*(self.maxPushDist - pushDist) + c1*(np.exp(-(pushDist**2)/c2) + np.exp(-(pushDist**2)/c3))
                pushRew = max(pushRew, 0)
            else:
                pushRew = 0
            reward = reachRew + pushRew
            return [reward, reachRew, reachDist, pushRew, pushDist, None, None, None]

        def compute_reward_pick_place(actions, obs):
            del obs

            reachDist = np.linalg.norm(objPos - fingerCOM)
            placingDist = np.linalg.norm(objPos - goal)
            assert np.all(goal == self._get_site_pos('goal_pick_place'))

            def reachReward():
                reachRew = -reachDist
                reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
                zRew = np.linalg.norm(fingerCOM[-1] - self.init_fingerCOM[-1])

                if reachDistxy < 0.05:
                    reachRew = -reachDist
                else:
                    reachRew =  -reachDistxy - 2*zRew

                #incentive to close fingers when reachDist is small
                if reachDist < 0.05:
                    reachRew = -reachDist + max(actions[-1],0)/50

                return reachRew , reachDist

            def pickCompletionCriteria():
                tolerance = 0.01
                if objPos[2] >= (heightTarget- tolerance):
                    return True
                else:
                    return False

            if pickCompletionCriteria():
                self.pickCompleted = True


            def objDropped():
                return (objPos[2] < (self.objHeight + 0.005)) and (placingDist >0.02) and (reachDist > 0.02)
                # Object on the ground, far away from the goal, and from the gripper
                # Can tweak the margin limits

            def orig_pickReward():
                hScale = 100
                if self.pickCompleted and not(objDropped()):
                    return hScale*heightTarget
                elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
                    return hScale* min(heightTarget, objPos[2])
                else:
                    return 0

            def placeReward():
                c1 = 1000
                c2 = 0.01
                c3 = 0.001
                cond = self.pickCompleted and (reachDist < 0.1) and not(objDropped())
                if cond:
                    placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
                    placeRew = max(placeRew, 0)
                    return [placeRew, placingDist]
                else:
                    return [0, placingDist]

            reachRew, reachDist = reachReward()
            pickRew = orig_pickReward()
            placeRew , placingDist = placeReward()
            assert ((placeRew >= 0) and (pickRew >= 0))
            reward = reachRew + pickRew + placeRew

            return [reward, reachRew, reachDist, None, None, pickRew, placeRew, placingDist]

        if self.task_type == 'reach':
            return compute_reward_reach(actions, obs)
        elif self.task_type == 'push':
            return compute_reward_push(actions, obs)
        elif self.task_type == 'pick_place':
            return compute_reward_pick_place(actions, obs)
        else:
            raise NotImplementedError

    ########### kuka sim2real interface ###########

    def _set_task_inner(self, *, task_type, **kwargs):
        super()._set_task_inner(**kwargs)
        self.task_type = task_type

        # we only do one task from [pick_place, reach, push]
        # per instance of SawyerReachPushPickPlaceEnv.
        # Please only set task_type from constructor.
        if self.task_type == 'pick_place':
            self.goal = np.array([0.65, 0.1, 0.2])
        elif self.task_type == 'reach':
            self.goal = np.array([0.65, -0.1, 0.2])
        elif self.task_type == 'push':
            self.goal = np.array([0.65, 0.1, 0.02])
        else:
            raise NotImplementedError

    @property
    def model_name(self):
        return full_v1_path_for('sawyer_xyz/sawyer_reach_push_pick_and_place_kuka.xml')

    @_assert_task_is_set
    def step(self, action):
        print("------ sawyer reach push pick place step ------")
        print(action)
        reward, reachDist, pushDist, pickRew, placingDist = 0, 0, 0, 0, 0
        if not self.sim2real:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
            ob = super().step(action)
            print("not sim2real step obs:", ob)
            reward, _, reachDist, _, pushDist, pickRew, _, placingDist = self.compute_reward(action, ob)
        else:
            print("##########################")
            ob = self.kuka_step(action)
            reward, _, reachDist, _, pushDist, pickRew, _, placingDist = self.kuka_compute_reward(action, ob)


        self.curr_path_length +=1
        print("-----------------------------------------------")
        print(f'task type: {self.task_type}')
        print(f'reward: {reward}')
        print("-----------------------------------------------\n")

        goal_dist = placingDist if self.task_type == 'pick_place' else pushDist

        if self.task_type == 'reach':
            success = float(reachDist <= 0.05)
        else:
            success = float(goal_dist <= 0.07)

        info = {
            'reachDist': reachDist,
            'pickRew': pickRew,
            'epRew': reward,
            'goalDist': goal_dist,
            'success': success
        }

        return ob, reward, False, info

    @property
    def _target_site_config(self):
        far_away = np.array([10., 10., 10.])
        return [
            ('goal_' + t, self._target_pos if t == self.task_type else far_away)
            for t in self.task_types
        ]

    def _get_pos_objects(self):
        return self.data.get_geom_xpos('objGeom')

    def adjust_initObjPos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not aligned
        # If this is not done, the object could be initialized in an extreme position
        diff = self.get_body_com('obj')[:2] - self.data.get_geom_xpos('objGeom')[:2]
        adjustedPos = orig_init_pos[:2] + diff

        # The convention we follow is that body_com[2] is always 0, and geom_pos[2] is the object height
        return [adjustedPos[0], adjustedPos[1],self.data.get_geom_xpos('objGeom')[-1]]

    def reset_model(self):
        if not self.sim2real:
            print("reset model")
            print("hand_init_pos", self.hand_init_pos)
            self._reset_hand()
            self._target_pos = self._get_state_rand_vec()
            self.obj_init_pos = self.adjust_initObjPos(self.init_config['obj_init_pos'])
            self.obj_init_angle = self.init_config['obj_init_angle']
            self.objHeight = self.data.get_geom_xpos('objGeom')[2]
            self.heightTarget = self.objHeight + self.liftThresh

            if self.random_init:
                goal_pos = self._get_state_rand_vec()
                self._target_pos = goal_pos[3:]
                while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
                    goal_pos = self._get_state_rand_vec()
                    self._target_pos = goal_pos[3:]
                if self.task_type == 'push':
                    self._target_pos = np.concatenate((goal_pos[-3:-1], [self.obj_init_pos[-1]]))
                    self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
                else:
                    self._target_pos = goal_pos[-3:]
                    self.obj_init_pos = goal_pos[:3]
            self._target_pos = self.kuka_goal
            print("initial hand pos:", self.get_endeff_pos())
            print("initial goal pos:", self._target_pos)
            self._set_obj_xyz(self.obj_init_pos)
            self.maxReachDist = np.linalg.norm(self.init_fingerCOM - np.array(self._target_pos))
            self.maxPushDist = np.linalg.norm(self.obj_init_pos[:2] - np.array(self._target_pos)[:2])
            self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._target_pos)) + self.heightTarget
            self.target_rewards = [1000*self.maxPlacingDist + 1000*2, 1000*self.maxReachDist + 1000*2, 1000*self.maxPushDist + 1000*2]

            if self.task_type == 'reach':
                idx = 1
            elif self.task_type == 'push':
                idx = 2
            elif self.task_type == 'pick_place':
                idx = 0
            else:
                raise NotImplementedError

            self.target_reward = self.target_rewards[idx]
            self.num_resets += 1
            print("target rewards", self.target_rewards)
            return self._get_obs()
        else:
            self.init_fingerCOM = self.kuka_env.getCurrentFrame() + np.array([0.0, 0.0, self.fingerHeight])
            self.maxReachDist = np.linalg.norm(self.init_fingerCOM - np.array(self.kuka_get_pos_goal()))
            self.maxPushDist = np.linalg.norm(self.kuka_get_pos_obj()[:2] - np.array(self.kuka_get_pos_goal())[:2])
            self.maxPlacingDist = np.linalg.norm(np.array([self.kuka_get_pos_obj()[0], self.kuka_get_pos_obj()[1], self.heightTarget]) - np.array(self.kuka_get_pos_goal())) + self.heightTarget
            self.target_rewards = [1000*self.maxPlacingDist + 1000*2, 1000*self.maxReachDist + 1000*2, 1000*self.maxPushDist + 1000*2]

            if self.task_type == 'reach':
                idx = 1
            elif self.task_type == 'push':
                idx = 2
            elif self.task_type == 'pick_place':
                idx = 0
            else:
                raise NotImplementedError

            self.target_reward = self.target_rewards[idx]
            print("kuka target rewards", self.target_rewards)
            self.num_resets += 1

            return self.kuka_get_obs()

    def _reset_hand(self):
        super()._reset_hand(10)
        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def compute_reward(self, actions, obs):

        objPos = obs[3:6]

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        heightTarget = self.heightTarget
        goal = self._target_pos

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("endeff pos", self.get_endeff_pos())
        print("fingerCOM", fingerCOM)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        def compute_reward_reach(actions, obs):
            del actions
            del obs

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            reachDist = np.linalg.norm(fingerCOM - goal)
            print("reachDist", reachDist)
            reachRew = c1*(self.maxReachDist - reachDist) + c1*(np.exp(-(reachDist**2)/c2) + np.exp(-(reachDist**2)/c3))
            print("reachRew", reachRew)
            reachRew = max(reachRew, 0)
            reward = reachRew
            print("reward", reward)
            return [reward, reachRew, reachDist, None, None, None, None, None]

        def compute_reward_push(actions, obs):
            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            del actions
            del obs

            assert np.all(goal == self._get_site_pos('goal_push'))
            reachDist = np.linalg.norm(fingerCOM - objPos)
            pushDist = np.linalg.norm(objPos[:2] - goal[:2])
            reachRew = -reachDist
            if reachDist < 0.05:
                pushRew = 1000*(self.maxPushDist - pushDist) + c1*(np.exp(-(pushDist**2)/c2) + np.exp(-(pushDist**2)/c3))
                pushRew = max(pushRew, 0)
            else:
                pushRew = 0
            reward = reachRew + pushRew
            return [reward, reachRew, reachDist, pushRew, pushDist, None, None, None]

        def compute_reward_pick_place(actions, obs):
            del obs

            reachDist = np.linalg.norm(objPos - fingerCOM)
            placingDist = np.linalg.norm(objPos - goal)
            assert np.all(goal == self._get_site_pos('goal_pick_place'))

            def reachReward():
                reachRew = -reachDist
                reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
                zRew = np.linalg.norm(fingerCOM[-1] - self.init_fingerCOM[-1])

                if reachDistxy < 0.05:
                    reachRew = -reachDist
                else:
                    reachRew =  -reachDistxy - 2*zRew

                #incentive to close fingers when reachDist is small
                if reachDist < 0.05:
                    reachRew = -reachDist + max(actions[-1],0)/50

                return reachRew , reachDist

            def pickCompletionCriteria():
                tolerance = 0.01
                if objPos[2] >= (heightTarget- tolerance):
                    return True
                else:
                    return False

            if pickCompletionCriteria():
                self.pickCompleted = True


            def objDropped():
                return (objPos[2] < (self.objHeight + 0.005)) and (placingDist >0.02) and (reachDist > 0.02)
                # Object on the ground, far away from the goal, and from the gripper
                # Can tweak the margin limits

            def orig_pickReward():
                hScale = 100
                if self.pickCompleted and not(objDropped()):
                    return hScale*heightTarget
                elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
                    return hScale* min(heightTarget, objPos[2])
                else:
                    return 0

            def placeReward():
                c1 = 1000
                c2 = 0.01
                c3 = 0.001
                cond = self.pickCompleted and (reachDist < 0.1) and not(objDropped())
                if cond:
                    placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
                    placeRew = max(placeRew, 0)
                    return [placeRew, placingDist]
                else:
                    return [0, placingDist]

            reachRew, reachDist = reachReward()
            pickRew = orig_pickReward()
            placeRew , placingDist = placeReward()
            assert ((placeRew >= 0) and (pickRew >= 0))
            reward = reachRew + pickRew + placeRew

            return [reward, reachRew, reachDist, None, None, pickRew, placeRew, placingDist]

        if self.task_type == 'reach':
            return compute_reward_reach(actions, obs)
        elif self.task_type == 'push':
            return compute_reward_push(actions, obs)
        elif self.task_type == 'pick_place':
            return compute_reward_pick_place(actions, obs)
        else:
            raise NotImplementedError
