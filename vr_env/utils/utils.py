import os
from pathlib import Path
import re
import subprocess
import time
from typing import Union

import git
import numpy as np
import quaternion


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed


class FpsController:
    def __init__(self, freq):
        self.loop_time = (1.0 / freq) * 10 ** 9
        self.prev_time = time.time_ns()

    def step(self):
        current_time = time.time_ns()
        delta_t = current_time - self.prev_time
        if delta_t < self.loop_time:
            nano_sleep(self.loop_time - delta_t)
        self.prev_time = time.time_ns()


def xyzw_to_wxyz(arr):
    """
    Convert quaternions from pyBullet to numpy.
    """
    return [arr[3], arr[0], arr[1], arr[2]]


def wxyz_to_xyzw(arr):
    """
    Convert quaternions from numpy to pyBullet.
    """
    return [arr[1], arr[2], arr[3], arr[0]]


def file_to_split_str(filepath):
    """
    Reads a file as string and returns and returns the str.split() content.
    """
    with open(filepath, "r") as file:
        data = file.read()
        data = data.split()
    return data


def nano_sleep(time_ns):
    """
    Spinlock style sleep function. Burns cpu power on purpose
    equivalent to time.sleep(time_ns / (10 ** 9)).

    Should be more precise, especially on Windows.

    Args:
        time_ns: time to sleep in ns

    Returns:

    """
    wait_until = time.time_ns() + time_ns
    while time.time_ns() < wait_until:
        pass


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between_quaternions(q1, q2):
    """
    Returns the minimum rotation angle between to orientations expressed as quaternions
    quaternions use X,Y,Z,W convention
    """
    q1 = xyzw_to_wxyz(q1)
    q2 = xyzw_to_wxyz(q2)
    q1 = quaternion.from_float_array(q1)
    q2 = quaternion.from_float_array(q2)

    theta = 2 * np.arcsin(np.linalg.norm((q1 * q2.conjugate()).vec))
    return theta


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def avg(avg_list: list) -> float:
    """
    get avg over list. return 0 otherwise.
    Args:
        avg_list:

    Returns:

    """
    if len(avg_list) == 0:
        return 0
    return sum(avg_list) / len(avg_list)


def print_ok(msg):
    """
    Prints msg in green.

    Args:
        msg: the message

    Returns:
        None
    """
    print(f"{Color.OKGREEN}{msg}{Color.ENDC}")


def print_fail(msg):
    """
    Prints msg in red.

    Args:
        msg:

    Returns:
        None
    """
    print(f"{Color.FAIL}{msg}{Color.ENDC}")


class Color:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def get_git_commit_hash(repo_path: Path) -> str:
    repo = git.Repo(search_parent_directories=True, path=repo_path.parent)
    assert repo, "not a repo"
    changed_files = [item.a_path for item in repo.index.diff(None)]
    if changed_files:
        print("WARNING uncommitted modified files: {}".format(",".join(changed_files)))
    return repo.head.object.hexsha


class EglDeviceNotFoundError(Exception):
    """Raised when EGL device cannot be found"""


def get_egl_device_id(cuda_id: int) -> Union[int]:
    """
    >>> i = get_egl_device_id(0)
    >>> isinstance(i, int)
    True
    """
    assert isinstance(cuda_id, int), "cuda_id has to be integer"
    dir_path = Path(__file__).absolute().parents[2] / "egl_check"
    if not os.path.isfile(dir_path / "EGL_options.o"):
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print("Building EGL_options.o")
            subprocess.call(["bash", "build.sh"], cwd=dir_path)
        else:
            # In case EGL_options.o has to be built and multiprocessing is used, give rank 0 process time to build
            time.sleep(5)
    result = subprocess.run(["./EGL_options.o"], capture_output=True, cwd=dir_path)
    n = int(result.stderr.decode("utf-8").split(" of ")[1].split(".")[0])
    for egl_id in range(n):
        my_env = os.environ.copy()
        my_env["EGL_VISIBLE_DEVICE"] = str(egl_id)
        result = subprocess.run(["./EGL_options.o"], capture_output=True, cwd=dir_path, env=my_env)
        match = re.search(r"CUDA_DEVICE=[0-9]+", result.stdout.decode("utf-8"))
        if match:
            current_cuda_id = int(match[0].split("=")[1])
            if cuda_id == current_cuda_id:
                return egl_id
    raise EglDeviceNotFoundError


def angle_between_angles(a, b):
    diff = b - a
    return (diff + np.pi) % (2 * np.pi) - np.pi


def to_relative_action(actions, robot_obs, max_pos=0.02, max_orn=0.05):
    assert isinstance(actions, np.ndarray)
    assert isinstance(robot_obs, np.ndarray)

    rel_pos = actions[:3] - robot_obs[:3]
    rel_pos = np.clip(rel_pos, -max_pos, max_pos) / max_pos

    rel_orn = angle_between_angles(robot_obs[3:6], actions[3:6])
    rel_orn = np.clip(rel_orn, -max_orn, max_orn) / max_orn

    gripper = actions[-1:]
    return np.concatenate([rel_pos, rel_orn, gripper])


if __name__ == "__main__":
    import doctest

    doctest.testmod()
