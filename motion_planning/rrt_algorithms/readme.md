## Code explanation
### `rrt_star_2d_test.py`
- for a static obstacle, based on the RRT*, add an adaptive step for the new rand node, instead of a constant step;
- replace the straight line check between the goal and the closest point, by the DS modulation check.








## An improved realtime RRT*
### Algorithm structure
`Input`: $q_{a}$, $x_{obs}$, $q_{g}$\
Initialize tree $T$ by current robot state $q_{a}$\

While not reach the goal: \
&nbsp;&nbsp;&nbsp;&nbsp; Update distances and gradients for all nodes $d, \frac{\partial d}{\partial q} = g(q_{n}, x_{obs})$



