import logging
import math
import random
import sys
import time
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QSize, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QScrollArea, QHBoxLayout, QLineEdit, \
    QPushButton, QMessageBox, QCheckBox, QFileDialog
from pydash import find_index

from src.adg import build_adg, to_adg_key
from src.domain import MapConfig, MapReq, x_y_to_index, State, distance_of_two_points, \
    MapfResult, RobotTaskReq, RobotTask, Cell, FaOp
from src.ecbs import ECBS
from src.ui_cell import CellUi
from src.ui_robot import RobotWidget

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='mapf.log'
)

# width and height of map cell
cell_size = 24


def add_widget_with_label(layout, widget, label_text):
    """
    Add a widget with a label to the layout.
    """
    hbox = QHBoxLayout()
    label = QLabel(label_text)
    hbox.addWidget(label)
    hbox.addWidget(widget)
    layout.addLayout(hbox)


@dataclass
class Node:
    """
    A* node used by ui
    """
    state: State
    parent: Optional['Node']
    g: float
    h: float
    f: float


@dataclass
class RobotExePath:
    """
    Robot state during simulation.
    """
    s2Index: int  # the index of s2(target) state in the path
    adgKey: str  # s2 index to adg key
    timeStart: int  # the time start of the path
    timeEnd: int  # the time end of the path
    startOn: int  # the time point in mm of simulation start
    s1: State  # the start state
    s2: State  # the goal state
    x: int  # current position x
    y: int  # current position y
    head: int  # current robot head (degree)
    rotateDuration: int  # the time step after rotation done
    moveDuration: int  # the time step after movement done
    waitDuration: int  # the time step after waiting done
    p: float  # step progress 0 ~ 1
    holding: bool  # waiting for other robot according to ADG


@dataclass
class RobotPosition:
    x: int
    y: int
    head: int


@dataclass
class LowSearchCell:
    color: QColor
    order: int
    tool_tip: str = ""


class MapfUi:
    def __init__(self):
        self.map_config = MapConfig()
        self.map_req = MapReq(self.map_config, {})

        self.robot_colors: dict[str, QColor] = {}

        self.target_to_robot: dict[int, str] = {}

        # the plan result
        self.plan: Optional[MapfResult] = None

        self.robot_widgets: dict[str, RobotWidget] = {}  # by robot name

        self.adg_nodes: dict[str, list[str]] = {}
        self.finished_adg_nodes: set[str] = set()

        self.simulation = False
        self.sim_speed = 4
        self.current_time = 0  # in mm
        self.stepDurationVar = .1
        self.sim_robots: dict[str, RobotExePath] = {}

        self.low_search_index: dict[int, LowSearchCell] = {}

        # 创建一个垂直布局
        main_layout = QVBoxLayout()
        self.main_layout = main_layout

        # 创建一个滚动区域
        scroll_area = QScrollArea()

        # 创建一个包含大量内容的部件
        content_widget = QWidget()
        content_layout = QVBoxLayout()

        button_layout = QHBoxLayout()

        save_map_btn = QPushButton('Save Map')
        button_layout.addWidget(save_map_btn)
        save_map_btn.clicked.connect(self.save_map)

        open_map_btn = QPushButton('Open Map')
        button_layout.addWidget(open_map_btn)
        open_map_btn.clicked.connect(self.open_map)

        random_obstacle_btn = QPushButton('Random Obstacles')
        button_layout.addWidget(random_obstacle_btn)
        random_obstacle_btn.clicked.connect(self.random_obstacles)

        clear_obstacles_btn = QPushButton('Clear Obstacles')
        button_layout.addWidget(clear_obstacles_btn)
        clear_obstacles_btn.clicked.connect(self.clear_obstacles)

        init_targets_btn = QPushButton('Random Targets')
        button_layout.addWidget(init_targets_btn)
        init_targets_btn.clicked.connect(self.random_targets)

        resolve_btn = QPushButton('Resolve')
        button_layout.addWidget(resolve_btn)
        resolve_btn.clicked.connect(self.resolve)

        sim_btn = QPushButton('Start Sim')
        self.sim_btn = sim_btn
        button_layout.addWidget(sim_btn)
        sim_btn.clicked.connect(self.toggle_sim)

        content_layout.addLayout(button_layout)

        input_line_1 = QHBoxLayout()

        self.robot_num_edit = QLineEdit(str(self.map_config.robotNum))
        add_widget_with_label(input_line_1, self.robot_num_edit, 'Robot Num:')

        self.map_dim_x_edit = QLineEdit(str(self.map_config.mapDimX))
        add_widget_with_label(input_line_1, self.map_dim_x_edit, 'Map Dim X:')

        self.map_dim_y_edit = QLineEdit(str(self.map_config.mapDimY))
        add_widget_with_label(input_line_1, self.map_dim_y_edit, 'Map Dim Y:')

        self.obstacle_ratio_edit = QLineEdit(str(self.map_config.obstacleRatio))
        add_widget_with_label(input_line_1, self.obstacle_ratio_edit, 'Obstacle Ratio:')

        self.toggle_obstacle_cb = QCheckBox("Toggle Obstacle")
        add_widget_with_label(input_line_1, self.toggle_obstacle_cb, '')

        content_layout.addLayout(input_line_1, 0)

        input_line_2 = QHBoxLayout()

        self.w_edit = QLineEdit(str(self.map_config.w))
        add_widget_with_label(input_line_2, self.w_edit, 'W:')

        self.target_num_edit = QLineEdit(str(self.map_config.targetNum))
        add_widget_with_label(input_line_2, self.target_num_edit, 'Target Num:')

        self.goal_stop_times_edit = QLineEdit(str(self.map_config.goalStopTimes))
        add_widget_with_label(input_line_2, self.goal_stop_times_edit, 'Goal Stop Times:')

        self.sim_speed_edit = QLineEdit(str(self.sim_speed))
        add_widget_with_label(input_line_2, self.sim_speed_edit, 'Sim Speed:')

        content_layout.addLayout(input_line_2, 0)

        input_line_3 = QHBoxLayout()

        self.tasks_edit = QLineEdit()
        add_widget_with_label(input_line_3, self.tasks_edit, 'Tasks:')

        self.result_edit = QLineEdit()
        self.result_edit.setReadOnly(True)
        add_widget_with_label(input_line_3, self.result_edit, 'Result:')

        content_layout.addLayout(input_line_3, 0)

        input_line_4 = QHBoxLayout()

        content_layout.addLayout(input_line_4)
        self.load_low_search_btn = QPushButton('Load Low Search')
        input_line_4.addWidget(self.load_low_search_btn)
        self.load_low_search_btn.clicked.connect(self.load_low_search)

        ########

        self.init_obstacles()
        self.reset_robot_colors()

        map_grid = QWidget()
        self.map_grid = map_grid

        self.map_cells: list[CellUi] = []
        self.rebuild_map_cells()

        content_layout.addWidget(map_grid, 1)

        content_layout.addStretch(100000)

        content_widget.setLayout(content_layout)

        # 将内容部件设置到滚动区域中
        scroll_area.setWidget(content_widget)
        scroll_area.setWidgetResizable(True)
        self.scroll_area = scroll_area

        # 将滚动区域添加到主布局中
        main_layout.addWidget(scroll_area)

        # 创建主窗口
        main_window = QWidget()
        self.main_window = main_window

        main_window.setWindowTitle('MAPF DEV')
        # main_window.setGeometry(100, 100, 400, 300)

        # 将布局设置到主窗口
        main_window.setLayout(main_layout)

        self.timer = QTimer(main_window)
        self.timer.timeout.connect(self.sim_loop)
        self.timer.start(int(1000 / 10))

    def do_inputs(self):
        self.map_config.robotNum = int(self.robot_num_edit.text())
        self.map_config.mapDimX = int(self.map_dim_x_edit.text())
        self.map_config.mapDimY = int(self.map_dim_y_edit.text())
        self.map_config.obstacleRatio = float(self.obstacle_ratio_edit.text())
        self.map_config.w = float(self.w_edit.text())
        self.map_config.targetNum = int(self.target_num_edit.text())
        self.map_config.goalStopTimes = int(self.goal_stop_times_edit.text())

        tasks_str = self.tasks_edit.text()
        task_str_list = tasks_str.split(",")
        tasks: dict[str, RobotTaskReq] = {}
        for task_str in task_str_list:
            parts = task_str.split(":")
            if len(parts) != 3:
                continue
            robot_name = parts[0].strip()
            from_index = parts[1].strip()
            to_index = parts[2].strip()

            if not (robot_name and from_index and to_index):
                continue
            tasks[robot_name] = RobotTaskReq(int(from_index), int(to_index))
            self.target_to_robot[int(to_index)] = robot_name
        self.map_req.tasks = tasks

        self.sim_speed = int(self.sim_speed_edit.text())

        self.reset_robot_colors()

    def reset_robot_colors(self):
        self.robot_colors = {}
        for i in range(self.map_config.robotNum):
            hue = i * 137.508  # use golden angle approximation
            color = QColor()
            color.setHsl(int(hue), int(255 * .7), int(255 * .5))
            self.robot_colors[str(i)] = color

    def init_obstacles(self):
        """
        只产生数据，不更新 UI
        """

        self.do_inputs()

        map_dim_x: int = self.map_config.mapDimX
        map_dim_y: int = self.map_config.mapDimY
        obstacle_ratio: float = self.map_config.obstacleRatio

        cell_num = map_dim_x * map_dim_y
        obstacle_num = round(cell_num * obstacle_ratio)
        cells = [0] * cell_num
        for i in range(cell_num):
            cells[i] = 1 if i < obstacle_num else 0
        random.shuffle(cells)
        obstacles = set()
        for i in range(cell_num):
            if cells[i]:
                obstacles.add(i)

        self.map_config.obstacles = obstacles
        self.map_req.tasks = {}
        self.target_to_robot = {}
        self.plan = None

    def random_obstacles(self):
        print("random_obstacles")
        self.init_obstacles()
        self.rebuild_map_cells()

    def clear_obstacles(self):
        self.do_inputs()

        self.map_config.obstacles = set()
        self.map_req.tasks = {}
        self.target_to_robot = {}
        self.plan = None

        self.rebuild_map_cells()

    def rebuild_map_cells(self):
        for cell in self.map_cells:
            cell.setParent(None)
            cell.deleteLater()

        self.map_cells = []

        x_n = self.map_config.mapDimX
        y_n = self.map_config.mapDimY
        print(f"rebuild_map_cells, x={x_n}, y={y_n}")

        self.map_grid.setFixedSize(QSize(x_n * (cell_size + 1), y_n * (cell_size + 1)))

        for x in range(x_n):
            for y in range(y_n):
                index = x_y_to_index(x, y, x_n)

                fill = QColor("#ccc")
                label = ""
                tool_tip = ""
                obstacle = index in self.map_config.obstacles
                if obstacle:
                    fill = QColor("#888")
                else:
                    if self.low_search_index:
                        c = self.low_search_index.get(index)
                        if c:
                            fill = c.color
                            label = str(c.order)
                            tool_tip = c.tool_tip
                    else:
                        robot = self.target_to_robot.get(index)
                        if robot is not None:
                            fill = self.robot_colors[robot]

                cell = CellUi(cell_size, index, x, y, fill, label, tool_tip,
                              lambda x, y: self.toggle_obstacle(x, y))
                self.map_cells.append(cell)
                cell.setParent(self.map_grid)
                cell.show()

        print("rebuild_map_cells, done")

    def random_targets(self):
        self.do_inputs()

        tasks: dict[str, RobotTaskReq] = {}
        target_to_robot: dict[int, str] = {}

        used_indexes: set[int] = set()
        config = self.map_config
        cell_num = config.mapDimX * config.mapDimY
        if config.robotNum * 2 >= cell_num - len(config.obstacles):
            QMessageBox.warning(self.main_window, "Warning", "No enough room.")

        for i in range(config.robotNum):
            from_index = -1
            retry_start = 0
            while from_index < 0 and retry_start < cell_num:
                retry_start += 1
                while from_index < 0:
                    n = random.randint(0, cell_num - 1)
                    if n in config.obstacles or n in used_indexes:
                        continue
                    used_indexes.add(n)
                    from_index = n

                s1 = State(
                    x=from_index % config.mapDimX,
                    y=math.floor(from_index / config.mapDimX),
                    head=0,
                    timeNum=0,
                    timeStart=-1,
                    timeEnd=-1
                )
                task = RobotTaskReq(from_index, 0)
                tasks[str(i)] = task

                bad_indexes: set[int] = set()
                retry_target = 0
                for ti in range(config.targetNum):
                    to_index = -1
                    while to_index < 0 and retry_target < cell_num:
                        retry_target += 1
                        n = math.floor(random.random() * cell_num)
                        if n in self.map_config.obstacles or n in used_indexes or n in bad_indexes:
                            continue
                        if n == from_index:
                            continue

                        s2 = State(x=n % config.mapDimX, y=math.floor(n / config.mapDimX),
                                   head=0, timeNum=0, timeStart=-1, timeEnd=-1)
                        if not self.test_path(i, s1, s2):
                            bad_indexes.add(n)
                            continue

                        used_indexes.add(n)
                        to_index = n
                        task.toIndex = to_index
                        # task.toStates.append(state_to_cell(s2))

                        target_to_robot[n] = str(i)

                        if to_index < 0:
                            from_index = -1  # 重新选起点
                            break

            if from_index < 0:
                QMessageBox.warning(self.main_window, "Warning", "No good start.")
                return

        self.map_req.tasks = tasks
        self.target_to_robot = target_to_robot
        self.plan = None

        print(f"tasks: {tasks}")

        self.update_tasks_edit()

        self.rebuild_map_cells()
        self.update_robots_ui()

    def test_path(self, robot_name: int, from_state: State, to_state: State) -> bool:
        """
        Return true if there is a path from from_state to to_state.
        """
        print(f"find path: {robot_name}, {from_state} {to_state}")
        if from_state.x == to_state.x and from_state.y == to_state.y:
            return True
        open_set: list[Node] = [Node(state=from_state, g=0, h=0, f=0, parent=None)]
        close_set: list[Node] = []
        expanded_count = 0
        while open_set:
            expanded_count += 1
            top = open_set.pop(0)
            print(f"expanded: [{expanded_count}]R={robot_name}|x={top.state.x}|y={top.state.y}|f={top.f}")
            if top.state.x == to_state.x and top.state.y == to_state.y:
                return True
            close_set.append(top)
            neighbors = self.get_neighbors(top.state)
            for n in neighbors:
                g = top.g + 1
                h = distance_of_two_points(n.x, n.y, to_state.x, to_state.y)
                f = g + h
                open_index = find_index(open_set, lambda node: node.state.x == n.x and node.state.y == n.y)
                if open_index >= 0 and g < open_set[open_index].g:
                    open_set.pop(open_index)
                    open_index = -1
                close_index = find_index(close_set, lambda node: node.state.x == n.x and node.state.y == n.y)
                if close_index >= 0 and g < close_set[close_index].g:
                    close_set.pop(close_index)
                    close_index = -1
                if open_index < 0 and close_index < 0:
                    new_node = Node(state=n, g=g, h=h, f=f, parent=top)
                    open_set.append(new_node)

            open_set = sorted(open_set, key=lambda node: node.f)
        return False

    def get_neighbors(self, state: State) -> list[State]:
        neighbors: list[State] = []
        self.add_neighbor(neighbors, state, 0, 1)
        self.add_neighbor(neighbors, state, 0, -1)
        self.add_neighbor(neighbors, state, 1, 0)
        self.add_neighbor(neighbors, state, -1, 0)

        return neighbors

    def add_neighbor(self, neighbors: list[State], state: State, dx: int, dy: int) -> None:
        if state.x + dx < 0 or state.x + dx >= self.map_req.config.mapDimX:
            return
        if state.y + dy < 0 or state.y + dy >= self.map_req.config.mapDimY:
            return
        if ((state.y + dy) * self.map_req.config.mapDimX + state.x + dx) in self.map_config.obstacles:
            return
        neighbors.append(
            State(
                x=state.x + dx, y=state.y + dy, head=0,
                timeNum=0, timeStart=-1, timeEnd=-1
            )
        )

    def resolve(self):
        tasks: dict[str, RobotTask] = {}
        for robot_name, t in self.map_req.tasks.items():
            c1 = self.index_to_cell(t.fromIndex)
            c2 = self.index_to_cell(t.toIndex)
            tasks[robot_name] = RobotTask(robot_name,
                                          fromState=c1,
                                          toStates=[c2],
                                          stopTimes=self.map_config.goalStopTimes)

        resolver = ECBS(
            w=self.map_config.w,
            map_dim_x=self.map_config.mapDimX,
            map_dim_y=self.map_config.mapDimY,
            obstacles=self.map_config.obstacles,
            tasks=tasks,
        )
        r = resolver.search()
        self.plan = r
        self.result_edit.setText(r.to_json())
        print("Plan: " + str(r))

        msg = "Success" if r.ok else f"Fail：{r.msg}"
        QMessageBox.information(self.main_window, '完成', msg)

    def toggle_sim(self):
        self.do_inputs()

        if self.simulation:
            self.stop_sim()
        else:
            if not (self.plan and self.plan.ok):
                QMessageBox.warning(self.main_window, "Warning", "No ok plan.")

            self.sim_robots = {}
            for robot_name, plan in self.plan.plans.items():
                self.sim_robots[robot_name] = self.build_robot_exe_path(robot_name, 1, plan.path[0], plan.path[1])

            self.simulation = True
            self.sim_btn.setText('Stop Sim')
            self.adg_nodes = build_adg(self.plan)
            self.finished_adg_nodes.clear()

    def stop_sim(self):
        self.simulation = False
        self.sim_btn.setText('Start Sim')

    def sim_loop(self):
        if not self.simulation:
            return

        self.do_inputs()

        now = round(time.time() * 1000)
        robot_names = self.plan.plans.keys()

        # 第一轮循环，先推下进度
        for robot_name in robot_names:
            sim_robot = self.sim_robots[robot_name]
            duration = sim_robot.rotateDuration + sim_robot.moveDuration + sim_robot.waitDuration
            if duration <= 0:
                duration = 1000
            time_pass = now - sim_robot.startOn
            print(f"duration={duration}, time_pass={time_pass}")
            sim_robot.p = time_pass / duration * self.sim_speed
            if sim_robot.p >= 1:
                sim_robot.p = 1
                sim_robot.holding = True
                print(f"done ADG node {sim_robot.adgKey}")
                self.finished_adg_nodes.add(sim_robot.adgKey)
                if sim_robot.timeEnd > self.current_time:
                    self.current_time = sim_robot.timeEnd

            rp = self.get_position(sim_robot)
            sim_robot.x = rp.x
            sim_robot.y = rp.y
            sim_robot.head = rp.head

        # 分配下一步
        all_done = True
        for robot_name in robot_names:
            sim_robot = self.sim_robots[robot_name]
            if sim_robot.p < 1:
                all_done = False
                continue
            path = self.plan.plans[robot_name].path
            if not path:
                continue
            next_index = sim_robot.s2Index + 1
            s1 = path[sim_robot.s2Index] if sim_robot.s2Index < len(path) else None
            s2 = path[next_index] if next_index < len(path) else None
            if not s1 or not s2:
                continue
            all_done = False
            adg_key = to_adg_key(robot_name, next_index)
            dependents = self.adg_nodes.get(adg_key)
            dependents_all_pass = True
            if dependents:
                for d in dependents:
                    if d not in self.finished_adg_nodes:
                        dependents_all_pass = False
                        break
            if dependents_all_pass:
                self.sim_robots[robot_name] = self.build_robot_exe_path(robot_name, next_index, s1, s2)
                print(f"release ADG node {sim_robot.adgKey}")

        self.update_robots_ui()

        if all_done:
            print(f"Sim done")
            self.stop_sim()
            return

    def build_robot_exe_path(self, robot_name: str, s2_index: int, s1: State, s2: State) -> RobotExePath:
        # 需要转的角度，初始，-270 ~ +270
        d_head = abs(s2.head - s1.head)
        # 270 改成 90
        if d_head > 180:
            d_head = 90
        d_head /= 90
        rotate_time_num = math.ceil(d_head)
        move_time_num = abs(s1.x - s2.x + s1.y - s2.y)
        wait_time_num = s2.timeNum - rotate_time_num - move_time_num
        # 每 90 度 1 秒
        rotate_duration = rotate_time_num * 1000 * (1 + random.random() * self.stepDurationVar)
        # 旋转，则移动为 0
        move_duration = move_time_num * 1000 * (1 + random.random() * self.stepDurationVar)
        return RobotExePath(
            s2Index=s2_index,
            adgKey=to_adg_key(robot_name, s2_index),
            timeStart=s1.timeStart or 0,  # 取开始点的
            timeEnd=s2.timeEnd or 0,  # 取结束点的
            startOn=round(time.time() * 1000),
            s1=s1, s2=s2,
            rotateDuration=round(rotate_duration),
            moveDuration=round(move_duration),
            waitDuration=wait_time_num * 1000,
            p=0,
            x=s1.x * (cell_size + 1),
            y=s1.y * (cell_size + 1),
            head=s1.head,
            holding=False
        )

    def get_position(self, sim_robot: RobotExePath) -> RobotPosition:
        s1 = sim_robot.s1
        s2 = sim_robot.s2
        p_rotate = 1
        p_move = 1
        time_pass = round(time.time() * 1000) - sim_robot.startOn
        if sim_robot.rotateDuration > 0:
            p_rotate = time_pass / sim_robot.rotateDuration * self.sim_speed
            if p_rotate > 1:
                p_rotate = 1
        if sim_robot.moveDuration > 0:
            p_move = (time_pass - sim_robot.rotateDuration * self.sim_speed) / sim_robot.moveDuration * self.sim_speed
            if p_move > 1:
                p_move = 1
            if p_move < 0:
                p_move = 0
        return RobotPosition(
            x=round((s1.x + p_move * (s2.x - s1.x)) * (cell_size + 1)),
            y=round((s1.y + p_move * (s2.y - s1.y)) * (cell_size + 1)),
            head=round(s1.head + (s2.head - s1.head) * p_rotate)
        )

    def update_robots_ui(self):
        for (robot_name, r_ui) in self.robot_widgets.items():
            r_ui.setParent(None)
            r_ui.deleteLater()

        self.robot_widgets = {}

        robot_positions: dict[str, RobotPosition] = {}
        if self.simulation and self.plan.ok:
            for robot_name, sim_robot in self.sim_robots.items():
                robot_positions[robot_name] = RobotPosition(x=sim_robot.x,
                                                            y=sim_robot.y,
                                                            head=sim_robot.head)
        else:
            for robot_name, task in self.map_req.tasks.items():
                from_cell = self.index_to_cell(task.fromIndex)
                robot_positions[robot_name] = RobotPosition(
                    x=from_cell.x * (cell_size + 1),
                    y=from_cell.y * (cell_size + 1),
                    head=0)

        ri = 0
        for (robot_name, p) in robot_positions.items():
            print(f"robot {robot_name} position: {p}")

            color = self.robot_colors[robot_name]
            r_ui = RobotWidget(robot_name, cell_size, p.x, p.y, p.head, color, self.map_grid)
            self.robot_widgets[robot_name] = r_ui
            r_ui.show()

            ri += 1

    def toggle_obstacle(self, x: int, y: int):
        print(f"toggle obstacle: {x}, {y}")
        index = x_y_to_index(x, y, self.map_config.mapDimX)
        if index in self.map_config.obstacles:
            self.map_config.obstacles.remove(index)
        else:
            self.map_config.obstacles.add(index)

        # 暂时总体重绘
        self.rebuild_map_cells()

    def save_map(self):
        file_path, _ = QFileDialog.getSaveFileName(self.main_window, '保存地图', '', 'JSON 文件 (*.map.json)')
        txt = self.map_req.to_json()
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(txt)
            print('写入成功')
        except Exception as e:
            print(f'写入文件时出错: {e}')

    def open_map(self):
        file_path, _ = QFileDialog.getOpenFileName(self.main_window, '选择文件', '', 'JSON 文件 (*.map.json)')
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.load_map(content)
            except Exception as e:
                print(f'打开文件时出错: {e}')

    def load_map(self, content):
        req: MapReq = MapReq.from_json(content)
        self.map_req = req
        self.map_config = self.map_req.config

        self.target_to_robot = {}
        for (robot_name, t) in req.tasks.items():
            self.target_to_robot[t.toIndex] = robot_name

        self.stop_sim()

        self.reset_robot_colors()

        self.rebuild_map_cells()

        self.update_robots_ui()

        self.update_tasks_edit()

        self.robot_num_edit.setText(str(self.map_config.robotNum))
        self.map_dim_x_edit.setText(str(self.map_config.mapDimX))
        self.map_dim_y_edit.setText(str(self.map_config.mapDimY))
        self.obstacle_ratio_edit.setText(str(self.map_config.obstacleRatio))
        self.w_edit.setText(str(self.map_config.w))
        self.target_num_edit.setText(str(self.map_config.targetNum))
        self.goal_stop_times_edit.setText(str(self.map_config.goalStopTimes))
        self.sim_speed_edit.setText(str(self.sim_speed))

    def update_tasks_edit(self):
        tasks = [f"{robot_name}:{task.fromIndex}:{task.toIndex}" for robot_name, task in self.map_req.tasks.items()]
        tasks_str = ", ".join(tasks)
        self.tasks_edit.setText(tasks_str)

    def index_to_cell(self, index: int) -> Cell:
        x = index % self.map_config.mapDimX
        y = math.floor(index / self.map_config.mapDimX)
        return Cell(x, y)

    def load_low_search(self):
        if self.low_search_index:
            self.load_low_search_btn.setText('Load Low Search')
            self.low_search_index = None
        else:
            self.load_low_search_btn.setText('Clear Low Search')
            self.stop_sim()

            file_path, _ = QFileDialog.getOpenFileName(self.main_window, 'Select log file', '', 'JSON file (*.json)')
            if file_path:
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        op: FaOp = FaOp.from_json(content)
                        self.low_search_index = {}
                        for es in op.expandedList:
                            parts = es.split("|")
                            index = int(parts[0])
                            cell_index = int(parts[2])
                            color = QColor("red")
                            alpha = float(index) / op.expandedCount * .8 + .2
                            color.setAlpha(round(255 * alpha))
                            tool_tip = es
                            self.low_search_index[cell_index] = LowSearchCell(color, index, tool_tip)
                        self.low_search_index[op.startIndex] = LowSearchCell(QColor("blue"), 0)
                        self.low_search_index[op.goalIndex] = LowSearchCell(QColor("green"), 0)
                except Exception as e:
                    print(f'打开文件时出错: {e}')
                    return
        self.rebuild_map_cells()


def main():
    app = QApplication(sys.argv)

    mapf_ui = MapfUi()
    mapf_ui.main_window.resize(960, 600)
    mapf_ui.main_window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
