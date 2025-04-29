from typing import Optional

from src.common import is_time_overlay
from src.domain import RobotConstraints, State, ConflictConstraints, VertexConstraint, EdgeConstraint


def count_conflicts_find_first_constraints(
        paths: dict[str, list[State]]) -> (int, Optional[ConflictConstraints]):
    """
    计算所有机器人的总冲突数，并返回时间上最早的一个冲突产生的约束
    """
    conflicts_count = 0
    first_conflict: Optional[ConflictConstraints] = None

    robot_names = list(paths.keys())

    for r1i in range(len(robot_names) - 1):
        robot1 = robot_names[r1i]
        path1 = paths[robot1]
        for r2i in range(r1i + 1, len(robot_names)):
            robot2 = robot_names[r2i]
            path2 = paths[robot2]
            # 顶点约束
            # 遍历两个机器人的每个状态，如果位置相同且时间交叠
            for s1 in path1:
                for s2 in path2:
                    if (s1.is_same_location(s2) and
                            is_time_overlay(s1.timeStart, s1.timeEnd, s2.timeStart, s2.timeEnd)):
                        conflicts_count += 1
                        if (first_conflict is None
                                or s1.timeStart < first_conflict.timeStart
                                or s2.timeStart < first_conflict.timeStart):
                            # 注意用另一个机器人的时间
                            first_conflict = ConflictConstraints(
                                min(s1.timeStart, s2.timeStart),
                                {
                                    robot1: VertexConstraint(
                                        timeStart=s2.timeStart, timeEnd=s2.timeEnd, x=s1.x, y=s1.y),
                                    robot2: VertexConstraint(
                                        timeStart=s1.timeStart, timeEnd=s1.timeEnd, x=s2.x, y=s2.y
                                    )
                                }
                            )
            # 边约束
            # 遍历两个机器人的所有移动 i -> i+1
            # 时间范围取占起点的开始时间到终点的结束时间
            for s1i in range(len(path1) - 1):
                s1a = path1[s1i]
                s1b = path1[s1i + 1]
                if s1a.is_same_location(s1b): continue
                for s2i in range(len(path2) - 1):
                    s2a = path2[s2i]
                    s2b = path2[s2i + 1]
                    if s2a.is_same_location(s2b): continue
                    if (s1a.is_same_location(s2b) and s1b.is_same_location(s2a) and
                            is_time_overlay(s1b.timeStart, s1b.timeEnd, s2b.timeStart, s2b.timeEnd)):
                        conflicts_count += 1
                        if (first_conflict is None
                                or s1a.timeStart < first_conflict.timeStart
                                or s2a.timeStart < first_conflict.timeStart):
                            # 注意用另一个机器人的时间，但边的起点和终点还是选自己的（起点终点不需要反转）
                            first_conflict = ConflictConstraints(
                                min(s1a.timeStart, s2a.timeStart),
                                {
                                    robot1: EdgeConstraint(
                                        timeStart=s2a.timeStart, timeEnd=s2b.timeEnd,
                                        x1=s1a.x, y1=s1a.y, x2=s1b.x, y2=s1b.y),
                                    robot2: EdgeConstraint(
                                        timeStart=s1a.timeStart, timeEnd=s1b.timeEnd,
                                        x1=s2a.x, y1=s2a.y, x2=s2b.x, y2=s2b.y)
                                }
                            )

    return conflicts_count, first_conflict


def count_robot_transition_conflicts(robot1: str, s1a: State, s1b: State, all_paths: dict[str, list[State]]) -> int:
    """
    机器人从 s1a 转移到新状态 s1b 与其他机器人的冲突
    """
    conflicts_count = 0
    for robot2, path2 in all_paths.items():
        if robot1 != robot2 and path2 is not None:
            for s2i, s2a in enumerate(path2):
                if (s1b.is_same_location(s2a) and
                        is_time_overlay(s1b.timeStart, s1b.timeEnd, s2a.timeStart, s2a.timeEnd)):
                    conflicts_count += 1
                if s2i < len(path2) - 1:
                    s2b = path2[s2i + 1]
                    if (s1a.is_same_location(s2b) and s1b.is_same_location(s2a) and
                            # TODO s1a.timeStart vs s1b.timeStart
                            is_time_overlay(s1b.timeStart, s1b.timeEnd, s2b.timeStart, s2b.timeEnd)):
                        conflicts_count += 1

    return conflicts_count


def find_first_state_in_path_by_time_interval(path: Optional[list[State]], t_start: int, t_end: int) -> Optional[State]:
    """
    在状态列表（路径）中找与时间段重叠的第一个状态，可能为空
    """
    if not (path and len(path)):
        return None
    for state in path:
        if is_time_overlay(state.timeStart, state.timeEnd, t_start, t_end):
            return state
    return None


def check_constraints_intersect(robot_name: str, cs1: RobotConstraints, cs2: RobotConstraints):
    """
    两个约束集有交集
    """
    for vc in cs1.vertexConstraints:
        if vc in cs2.vertexConstraints:
            raise Exception(f"Constraints already exist for robot {robot_name}: {vc}")

    for ec in cs1.edgeConstraints:
        if ec in cs2.edgeConstraints:
            raise Exception(f"Constraints already exist for robot {robot_name}: {ec}")
