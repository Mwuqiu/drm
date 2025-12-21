import os
import numpy as np
import logging
import matplotlib.pyplot as plt    
from scipy.spatial.transform import Rotation as R


class Expression:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class UnaryAtomic(Expression):
    def __init__(self):
        pass

    def __call__(self, arg1):
        raise NotImplementedError


class BinaryAtomic(Expression):
    def __init__(self):
        pass

    def __call__(self, arg1, arg2):
        raise NotImplementedError


class MultiarayAtomic(Expression):
    def __init__(self):
        pass

    def __call__(self, *args):
        raise NotImplementedError


class TruePredicateFn(MultiarayAtomic):
    def __init__(self):
        super().__init__()

    def __call__(self, *args):
        return True


class FalsePredicateFn(MultiarayAtomic):
    def __init__(self):
        super().__init__()

    def __call__(self, *args):
        return False


class InContactPredicateFn(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg1.check_contact(arg2)


class In(BinaryAtomic):
    def __call__(self, arg1, arg2):
        is_contained = arg2.check_contain(arg1)
        is_in_contact = arg2.check_contact(arg1)
        final_result = is_contained and is_in_contact       
        # logging.info(f"is_contained : {is_contained}, is_in_contact : {is_in_contact}")
        return final_result


class On(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg2.check_ontop(arg1)


class Up(BinaryAtomic):
    def __call__(self, arg1):
        return arg1.get_geom_state()["pos"][2] >= 1.0


class Stack(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return (
            arg1.check_contact(arg2)
            and arg2.check_contain(arg1)
            and arg1.get_geom_state()["pos"][2] > arg2.get_geom_state()["pos"][2]
        )


class PrintJointState(UnaryAtomic):
    """This is a debug predicate to allow you print the joint values of the object you care"""

    def __call__(self, arg):
        print(arg.get_joint_state())
        return True


class Open(UnaryAtomic):
    def __call__(self, arg):
        return arg.is_open()


class Close(UnaryAtomic):
    def __call__(self, arg):
        return arg.is_close()


class TurnOn(UnaryAtomic):
    def __call__(self, arg):
        return arg.turn_on()


class TurnOff(UnaryAtomic):
    def __call__(self, arg):
        return arg.turn_off()


class IsOnBottomOf(BinaryAtomic):
    """
    A dedicated predicate that only checks if the bottom of arg1 is vertically 
    aligned with the bottom of arg2. It assumes arg1 and arg2 are physical objects.
    This version uses get_geom_xpos for more reliable position data.
    """
    def __call__(self, arg1, arg2):
        env = arg1.env
        sim = env.sim

        item_obj = env.get_object(arg1.object_name)
        item_geom_name = item_obj.contact_geoms[0]
        item_geom_id = sim.model.geom_name2id(item_geom_name)
        item_geom_pos = sim.data.get_geom_xpos(item_geom_name)
        item_geom_size = sim.model.geom_size[item_geom_id]
        item_half_height = item_geom_size[2]
        item_bottom_z = item_geom_pos[2] - item_half_height

        container_obj = env.get_object(arg2.object_name)
        container_geom_name = container_obj.contact_geoms[0]
        container_geom_id = sim.model.geom_name2id(container_geom_name)
        container_geom_pos = sim.data.get_geom_xpos(container_geom_name)
        container_geom_size = sim.model.geom_size[container_geom_id]
        container_half_height = container_geom_size[2]
        container_bottom_z = container_geom_pos[2] - container_half_height

        height_difference = abs(item_bottom_z - container_bottom_z)
        
        if arg2.object_name == "wooden_tray_1":
            height_threshold = 0.04
        elif arg2.object_name == "basket_1" and arg1.object_name == "ketchup_1":
            height_threshold = 0.09
        elif arg2.object_name == "basket_1" and arg1.object_name == "salad_dressing_1":
            height_threshold = 0.195
        elif arg2.object_name == "basket_1" and arg1.object_name == "alphabet_soup_1":
            height_threshold = 0.140
        elif arg2.object_name == "basket_1" and arg1.object_name == "bbq_sauce_1":
            height_threshold = 0.11
        elif arg2.object_name == "basket_1" and arg1.object_name == "tomato_sauce_1":
            height_threshold = 0.140
        elif arg2.object_name == "basket_1" and arg1.object_name == "milk_1":
            height_threshold = 0.09
        elif arg2.object_name == "basket_1" and arg1.object_name == "orange_juice_1":
            height_threshold = 0.09
        else:
            height_threshold = 0.0

        is_vertically_on_bottom = height_difference < height_threshold        
        # logging.info(f"IsOnBottomOf Check: Item Bottom Z: {item_bottom_z:.4f}, Container Bottom Z: {container_bottom_z:.4f}, Diff: {height_difference:.4f} -> {is_vertically_on_bottom}")
        return is_vertically_on_bottom


class IsGrasping(UnaryAtomic):

    def _check_contact_between_geoms(self, sim, geoms1, geoms2):
        for contact in sim.data.contact:
            geom1_name = sim.model.geom_id2name(contact.geom1)
            geom2_name = sim.model.geom_id2name(contact.geom2)

            if (geom1_name in geoms1 and geom2_name in geoms2) or \
               (geom1_name in geoms2 and geom2_name in geoms1):
                return True 
        return False 

    def __call__(self, arg1):
        env = arg1.env
        sim = env.sim

        real_object = env.get_object(arg1.object_name)
        object_geoms = real_object.contact_geoms

        gripper_geom_groups = {
            "left_fingerpad": [
                'gripper0_finger1_pad_collision',
                'gripper0_finger1_collision'
            ],
            "right_fingerpad": [
                'gripper0_finger2_pad_collision',
                'gripper0_finger2_collision'
            ]
        }

        left_contact = self._check_contact_between_geoms(sim, gripper_geom_groups["left_fingerpad"], object_geoms)
        right_contact = self._check_contact_between_geoms(sim, gripper_geom_groups["right_fingerpad"], object_geoms)
        is_grasping = left_contact and right_contact
        # logging.info(f"Grasp check: Left finger contact: {left_contact}, Right finger contact: {right_contact}. --> Final Result: {is_grasping}")
        return is_grasping


class PositionUnchanged(UnaryAtomic):
    _current_episode_index = -1

    @classmethod
    def start_new_episode(cls, episode_index):
        if cls._current_episode_index != episode_index:
            logging.warning(f"A new episode has begun (index {episode_index}). Predicate states will be reset on their next use.")
            cls._current_episode_index = episode_index

    def __init__(self, stability_threshold=5):
        super().__init__()
        self.STABILITY_THRESHOLD = stability_threshold
        self.official_initial_positions = {}
        self.stability_counters = {}
        self.instance_episode_index = -1

    def _reset_internal_state(self):
        self.official_initial_positions = {}
        self.stability_counters = {}
        logging.info(f"Instance of PositionUnchanged has reset its state for episode {self._current_episode_index}.")

    def __call__(self, arg1):
        if self.instance_episode_index != PositionUnchanged._current_episode_index:
            self._reset_internal_state()
            self.instance_episode_index = PositionUnchanged._current_episode_index

        object_name = arg1.object_name
        current_position = arg1.get_geom_state()["pos"]

        if object_name in self.official_initial_positions:
            initial_position = self.official_initial_positions[object_name]
            is_unchanged = np.allclose(current_position, initial_position, atol=1e-3)

            if not is_unchanged:
                distance = np.linalg.norm(current_position - initial_position)
                logging.warning(f"--- ❗ POSITION CHANGED for '{object_name}' ---")
                logging.warning(f"  - Initial Locked Position: {np.round(initial_position, 4)}")
                logging.warning(f"  - Current Position:        {np.round(current_position, 4)}")
                logging.warning(f"  - Distance Delta:          {distance:.6f}")

            return is_unchanged

        if object_name not in self.stability_counters:
            self.stability_counters[object_name] = {"count": 1, "pos": current_position.copy()}
            return True
        
        counter_data = self.stability_counters[object_name]
        if np.allclose(current_position, counter_data["pos"], atol=1e-3):
            counter_data["count"] += 1
        else: 
            counter_data["count"] = 1
            counter_data["pos"] = current_position.copy()
            
        if counter_data["count"] >= self.STABILITY_THRESHOLD:
            logging.info(f"--- '{object_name}' is STABLE. Locking initial position for episode {self.instance_episode_index}. ---")
            self.official_initial_positions[object_name] = counter_data["pos"]
            del self.stability_counters[object_name]
            
        return True


class IsPreciselyOn(BinaryAtomic):
    _current_episode_index = -1
    _episode_data = [] 

    @classmethod
    def start_new_episode(cls, episode_index, save_dir):
        if save_dir is not None and cls._episode_data and cls._current_episode_index != -1:
            cls.save_data(save_dir)
        cls._current_episode_index = episode_index
        cls._episode_data = []
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

    @classmethod
    def save_data(cls, save_dir):
        if not cls._episode_data:
            return
        filename = os.path.join(save_dir, f"episode_{cls._current_episode_index}_geom_pos.npy")
        try:
            data_to_save = np.array(cls._episode_data)
            np.save(filename, data_to_save)
            logging.info(f"Saved {len(data_to_save)} frames (Item+Container) to {filename}")
        except Exception as e:
            logging.error(f"Failed to save position data: {e}")

    PARAM_MAP = {
        "plate_1": {
            "offset": (0.0, 0.0),
            "thresholds": (0.01, 0.01),
            "angle_threshold": 5.0
        },
        "flat_stove_1_cook_region": {
            "offset": (0.0, 0.0),
            "thresholds": (0.01, 0.02),
            "angle_threshold": 5.0
        },
        "default": {
            "offset": (0.0, 0.0),
            "thresholds": (0.05, 0.05),
            "angle_threshold": 10.0
        }
    }

    def __call__(self, arg1, arg2):
        env = arg1.env
        sim = env.sim

        try:
            item_site_name = f"{arg1.object_name}_default_site"
            item_pos = sim.data.get_site_xpos(item_site_name).copy()

            container_name = arg2.object_name            
            container_obj = env.get_object(arg2.object_name)
            if hasattr(container_obj, 'contact_geoms'): 
                container_pos = sim.data.get_site_xpos(f"{container_name}_default_site").copy()
            else:
                container_pos = sim.data.get_site_xpos(container_name).copy()

            item_mat = sim.data.get_site_xmat(item_site_name).reshape(3, 3)
            r = R.from_matrix(item_mat)
            roll, pitch, _ = r.as_euler('xyz', degrees=True)

            config = self.PARAM_MAP.get(container_name, self.PARAM_MAP["default"])
            thresh_x, thresh_y = config["thresholds"]
            max_angle = config["angle_threshold"]
    
            self._episode_data.append([item_pos, container_pos]) 

            diff_x = abs(item_pos[0] - container_pos[0])
            diff_y = abs(item_pos[1] - container_pos[1])

            is_aligned = diff_x < thresh_x and diff_y < thresh_y     

            is_flat = abs(roll) < max_angle and abs(pitch) < max_angle

            return is_aligned and is_flat
                
        except Exception as e:
            logging.error(f"Error in IsPreciselyOn check: {e}")
            return False
    
    def debug_plot_object_anatomy(self, env, sim, obj_name):
        """
        这个函数会生成一张图，画出该物体所有部件的分布。
        """
        logging.info(f"--- Debugging Anatomy for: {obj_name} ---")
        
        # 1. 获取所有 Geoms (蓝色点)
        obj = env.get_object(obj_name)
        geom_names = obj.contact_geoms
        geom_positions = []
        for name in geom_names:
            pos = sim.data.get_geom_xpos(name).copy()
            geom_positions.append(pos)
            # 打印每个geom的坐标，方便你核对索引
            logging.info(f"Geom '{name}': {pos}")
        
        geom_positions = np.array(geom_positions)

        # 2. 获取相关 Sites (红色星)
        # 遍历所有 site，找到名字里包含 obj_name 的
        all_site_names = [sim.model.site_id2name(i) for i in range(sim.model.nsite)]
        related_sites = [s for s in all_site_names if obj_name in s]
        
        site_data = []
        for s in related_sites:
            pos = sim.data.get_site_xpos(s).copy()
            site_data.append((s, pos))
            logging.info(f"Site '{s}': {pos}")

        # 3. 开始绘图
        plt.figure(figsize=(10, 8))
        
        # 画 Geoms (蓝色)
        if len(geom_positions) > 0:
            plt.scatter(geom_positions[:, 0], geom_positions[:, 1], c='blue', alpha=0.6, label='Geoms (Body Parts)')
            # 给第0个geom (通常是把手) 打个特殊标记
            plt.text(geom_positions[0, 0], geom_positions[0, 1], "Geom[0]", fontsize=12, color='blue', fontweight='bold')
            
            # 画一个所有 Geoms 的平均中心 (绿色叉)
            centroid = np.mean(geom_positions, axis=0)
            plt.scatter(centroid[0], centroid[1], c='green', marker='x', s=200, linewidth=3, label='Calculated Centroid')
            logging.info(f"Calculated Geom Centroid: {centroid}")

        # 画 Sites (红色)
        for name, pos in site_data:
            plt.scatter(pos[0], pos[1], c='red', marker='*', s=200, label=f"Site: {name}")
            plt.text(pos[0], pos[1], name, fontsize=9, color='red')

        plt.title(f"Anatomy of {obj_name} (Top-Down View)")
        plt.xlabel("X Position (World)")
        plt.ylabel("Y Position (World)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal') # 保证比例一致，防止变形

        # 保存图片
        save_path = f"debug_anatomy_{obj_name}.png"
        plt.savefig(save_path)
        logging.info(f"Saved anatomy plot to {save_path}")
        plt.close()
        logging.info("-" * 30)


class IsOrientationAligned(BinaryAtomic):
    _current_episode_index = -1
    _episode_data = [] 
    _has_violated_constraint = False
    _active_constraints = {} 
    _current_call_count = 0

    @classmethod
    def start_new_episode(cls, episode_index, save_dir, constraints=None):
        if save_dir is not None and cls._episode_data and cls._current_episode_index != -1:
            cls.save_data(save_dir)

        cls._current_episode_index = episode_index
        cls._episode_data = []

        cls._has_violated_constraint = False 
        cls._active_constraints = constraints if constraints is not None else {}
        cls._current_call_count = 0
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

    @classmethod
    def save_data(cls, save_dir):
        if not cls._episode_data: return
        filename = os.path.join(save_dir, f"episode_{cls._current_episode_index}_geom_euler.npy")
        try:
            np.save(filename, np.array(cls._episode_data))
        except Exception: pass

    PARAM_MAP = {
        "default": {
            "tolerances": (5.0, 15.0, 5.0),
        }
    }

    def __init__(self, log_interval: int = 10, grace_period: int = 10):
        super().__init__()
        self.log_interval = log_interval
        self.GRACE_PERIOD = grace_period 

    def __call__(self, arg1, arg2):
        IsOrientationAligned._current_call_count += 1
        current_count = IsOrientationAligned._current_call_count
        
        env = arg1.env
        sim = env.sim
        try:
            item_obj = env.get_object(arg1.object_name)
            item_euler = R.from_matrix(sim.data.get_geom_xmat(item_obj.contact_geoms[0])).as_euler('xyz', degrees=True)

            container_name = arg2.object_name
            container_obj = env.get_object(container_name)
            if hasattr(container_obj, 'contact_geoms'):
                target_mat = sim.data.get_geom_xmat(container_obj.contact_geoms[0])
            else:
                target_mat = sim.data.get_site_xmat(container_name)
            target_euler = R.from_matrix(target_mat).as_euler('xyz', degrees=True)

            IsOrientationAligned._episode_data.append([item_euler, target_euler])

            def get_diff(a, b):
                d = abs(a - b)
                return min(d, 360 - d)

            roll_diff = get_diff(item_euler[0], target_euler[0])
            pitch_diff = get_diff(item_euler[1], target_euler[1])
            yaw_diff = get_diff(item_euler[2], target_euler[2])

            max_roll = IsOrientationAligned._active_constraints.get("max_roll_deviation")

            if (max_roll is not None) and (current_count > self.GRACE_PERIOD):
                if roll_diff > max_roll:
                    if not IsOrientationAligned._has_violated_constraint:
                        logging.warning(f"🚫 [Constraint Violation] Roll diff {roll_diff:.1f}° > {max_roll}°")
                    IsOrientationAligned._has_violated_constraint = True


            dynamic_tolerances = IsOrientationAligned._active_constraints.get("tolerances")
            if dynamic_tolerances is not None:
                tol_r, tol_p, tol_y = dynamic_tolerances
            else:
                config = self.PARAM_MAP.get(container_name, self.PARAM_MAP["default"])
                tol_r, tol_p, tol_y = config["tolerances"]
            
            is_aligned = (roll_diff < tol_r) and (pitch_diff < tol_p) and (yaw_diff < tol_y)
            return is_aligned and (not IsOrientationAligned._has_violated_constraint)

        except Exception as e:
            logging.error(f"Error: {e}")
            return False


class NotTooClose(BinaryAtomic):
    def __call__(self, arg1, arg2):
        MIN_DISTANCE = 0.1

        env = arg1.env
        sim = env.sim
        try:
            obj1_obj = env.get_object(arg1.object_name)
            obj1_geom_name = obj1_obj.contact_geoms[0]
            obj1_pos = sim.data.get_geom_xpos(obj1_geom_name)

            obj2_obj = env.get_object(arg2.object_name)
            obj2_geom_name = obj2_obj.contact_geoms[0]
            obj2_pos = sim.data.get_geom_xpos(obj2_geom_name)

            xy_distance = np.linalg.norm(obj1_pos[:2] - obj2_pos[:2])
            is_not_too_close = xy_distance > MIN_DISTANCE
            logging.info(
                f"--- NotTooClose Check ('{arg1.object_name}' vs '{arg2.object_name}') ---\n"
                f"  - XY Distance: {xy_distance:.4f}\n"
                f"  - Min Required Distance: {MIN_DISTANCE}\n"
                f"  -> Result: Not Too Close = {is_not_too_close}\n"
                f"----------------------------------------------------------"
            )
            return is_not_too_close

        except Exception as e:
            logging.error(f"Error in NotTooClose check: {e}")
            return False


class PitchRollStable(UnaryAtomic):
    _current_episode_index = -1

    @classmethod
    def start_new_episode(cls, episode_index):
        if cls._current_episode_index != episode_index:
            logging.warning(f"A new episode has begun (index {episode_index}). Predicate states will be reset on their next use.")
            cls._current_episode_index = episode_index

    def __init__(self, stability_threshold: int = 5, stability_atol_deg: float = 1.0, max_deviation_deg: float = 10.0):
        super().__init__()
        self.STABILITY_THRESHOLD = stability_threshold
        self.STABILITY_ATOL_DEG = stability_atol_deg
        self.MAX_DEVIATION_DEG = max_deviation_deg
        
        self.official_initial_orientations = {} 
        self.stability_counters = {}
        self.deviated_objects_in_episode = set() 
        
        self.instance_episode_index = -1

    def _reset_internal_state(self):
        """
        重置此实例的内部状态，用于新剧集。
        """
        self.official_initial_orientations = {}
        self.stability_counters = {}
        self.deviated_objects_in_episode = set()
        logging.info(f"Instance of PitchRollStable has reset its state for episode {self._current_episode_index}.")

    def _get_pr_diff(self, pr1: list, pr2: list) -> tuple:
        """
        辅助函数，计算两组 [roll, pitch] 之间的绝对差值，处理 360 度环绕。
        """
        roll_diff = abs(pr1[0] - pr2[0])
        roll_diff = min(roll_diff, 360.0 - roll_diff)
        
        pitch_diff = abs(pr1[1] - pr2[1])
        pitch_diff = min(pitch_diff, 360.0 - pitch_diff)
        
        return roll_diff, pitch_diff

    def __call__(self, arg1):
        """
        执行检测。
        """
        if self.instance_episode_index != PitchRollStable._current_episode_index:
            self._reset_internal_state()
            self.instance_episode_index = PitchRollStable._current_episode_index

        object_name = arg1.object_name
        
        if object_name in self.deviated_objects_in_episode:
            return False 

        try:
            env = arg1.env
            sim = env.sim
            item_obj = env.get_object(object_name)
            item_geom_name = item_obj.contact_geoms[0]
            item_rot_mat = sim.data.get_geom_xmat(item_geom_name)
            current_euler = R.from_matrix(item_rot_mat).as_euler('xyz', degrees=True)
            current_pr = [current_euler[0], current_euler[1]]
        
        except Exception as e:
            logging.error(f"Error in PitchRollStable getting orientation for {object_name}: {e}")
            return False 
        
        if object_name in self.official_initial_orientations:
            initial_pr = self.official_initial_orientations[object_name]
            roll_diff, pitch_diff = self._get_pr_diff(current_pr, initial_pr)

            is_stable = (roll_diff < self.MAX_DEVIATION_DEG) and (pitch_diff < self.MAX_DEVIATION_DEG)

            if not is_stable:
                logging.warning(f"--- ❗ PITCH/ROLL DEVIATION for '{object_name}' (Episode {self.instance_episode_index}) ---")
                logging.warning(f"  - Initial Locked (R, P): ({initial_pr[0]:.2f}, {initial_pr[1]:.2f})")
                logging.warning(f"  - Current (R, P):        ({current_pr[0]:.2f}, {current_pr[1]:.2f})")
                logging.warning(f"  - Diffs (R, P):          ({roll_diff:.2f}, {pitch_diff:.2f}) > Tol: {self.MAX_DEVIATION_DEG}")
                logging.warning(f"  - Object '{object_name}' is now permanently marked as UNSTABLE for this episode.")
                
                self.deviated_objects_in_episode.add(object_name)
                
                return False 
            return True 
        
        if object_name not in self.stability_counters:
            self.stability_counters[object_name] = {"count": 1, "euler_pr": current_pr}
            return True 
        
        counter_data = self.stability_counters[object_name]
        stable_sequence_pr = counter_data["euler_pr"]
        
        roll_diff, pitch_diff = self._get_pr_diff(current_pr, stable_sequence_pr)

        if (roll_diff < self.STABILITY_ATOL_DEG) and (pitch_diff < self.STABILITY_ATOL_DEG):
            counter_data["count"] += 1
        else:
            counter_data["count"] = 1
            counter_data["euler_pr"] = current_pr
        if counter_data["count"] >= self.STABILITY_THRESHOLD:
            logging.info(f"--- '{object_name}' PITCH/ROLL is STABLE. Locking initial orientation for episode {self.instance_episode_index}. ---")
            logging.info(f"  - Locked (R, P): ({counter_data['euler_pr'][0]:.2f}, {counter_data['euler_pr'][1]:.2f})")
            self.official_initial_orientations[object_name] = counter_data["euler_pr"]
            del self.stability_counters[object_name]
        return True 