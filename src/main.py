from src.domain import cell_to_state, Cell, RobotTask
from src.ecbs import ECBS
from src.fsa import FSA

# a_start = FSA(robot_name="A1", high_node_id=0, w=1.0,
#               map_dim_x=10, map_dim_y=2,
#               obstacles={1}
#               )
# pr = a_start.search(cell_to_state(Cell(0, 0)), [cell_to_state(Cell(4, 0))], 10)
# print(pr)
# print([n.desc_loc_head() for n in pr.path])

ecbs = ECBS(w=1.0, map_dim_x=10, map_dim_y=2, obstacles={2},
            tasks={"A1": RobotTask(name="A1", fromState=Cell(0, 0), toStates=[Cell(4, 0)]),
                   "A2": RobotTask(name="A2", fromState=Cell(1, 0), toStates=[Cell(5, 0)])
                   })
r = ecbs.search()
print("High expanded: " + str(ecbs.high_node_expanded))
print("Low expanded: " + str(ecbs.low_node_expanded))
print(r)

for robot_name, plan in r.plans.items():
    print(robot_name + ":" + str([n.desc_loc_head() for n in plan.path]))
