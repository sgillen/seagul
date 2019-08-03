import mujoco_py as mj


def find_contact_height(sim: mj.MjSim, iterations: int = 10) -> float:
    """ Find the height we should put our walker at so we are just barely in contact with the ground.

        Runs a binary search on height (assumed to be state[1][2]) such that the body with the minimum z coordinate
        on the ground is just barely in contact with the ground.


        Args:
            sim: the instantiated sim MjSim object you want to find the correct height for
            iterations: number of times to run the binary search
        Returns:
            just the height as a float, you'll need to set it yourself

    """

    state = sim.get_state()
    height_guess = state[1][1]
    step = height_guess*2   # upper range for how much we will search for

    for _ in range(iterations):
        if sim.data.ncon:   # ncon is the number of collisions
            height_guess += step
        else:
            height_guess -= step

        state[1][1] = height_guess
        sim.set_state(state)
        sim.forward()
        step /= 2

    return height_guess


# run the module to run some basic tests, no these aren't best practices but I don't care
if __name__ == '__main__':
    model_xml = open('walker.xml').read()
    model = mj.load_model_from_xml(model_xml)
    sim = mj.MjSim(model)
    viewer = mj.MjViewer(sim)

    height = find_contact_height(sim, iterations=20)
    state = sim.get_state()
    state[1][1] = height
    sim.set_state(state)
    sim.forward()
    print(height)

    # just manually interupt, since I'm lazy
    while(True):
        sim.forward()
        viewer.render()

