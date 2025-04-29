import math

from src.domain import Cell, State, TargetManyPlanResult


def x_y_to_index(x: int, y: int, map_dim_x: int) -> int:
    return y * map_dim_x + x


def distance_of_two_points(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def cell_to_state(c: Cell) -> State:
    return State(x=c.x, y=c.y, timeStart=0, timeEnd=0, timeNum=0, head=0)


def state_to_cell(s: State) -> Cell:
    return Cell(s.x, s.y)


def is_time_overlay(start1: int, end1: int, start2: int, end2: int):
    return start1 <= end2 and start2 <= end1


def robots_solution_to_robots_paths(solution: dict[str, TargetManyPlanResult]) -> dict[str, list[State]]:
    all_paths: dict[str, list[State]] = {}
    for robot_name, s in solution.items():
        if s.ok:
            all_paths[robot_name] = s.path
    return all_paths

