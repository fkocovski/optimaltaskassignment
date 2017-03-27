import randomstate.prng.pcg64 as pcg

from elements.workflow_process_elements import StartEvent, UserTask, XOR, DOR, COR

NUMBER_OF_USERS = 3
SERVICE_INTERVAL = 1
GENERATION_INTERVAL = 5
SIM_TIME = 100
BATCH_SIZE = 2
TASK_VARIABILITY = 0.2 * SERVICE_INTERVAL
WORKER_VARIABILITY = 0.2 * SERVICE_INTERVAL
SEED = 6


def create_files(name):
    file_policy = open(name, "w")
    file_policy.write("job,arrival,assigned,started,finished,user,task,task_name,token_id")
    for i in range(NUMBER_OF_USERS):
        file_policy.write(",user_{}_st".format(i + 1))
    file_policy.write("\n")
    return file_policy


def acquisition_process(env, policy, seed, generation_interval, accelerate, starting_generation, sim_time,
                        sigmoid_param):
    ut = UserTask(env, policy, "Setup Acquisition Offer", SERVICE_INTERVAL, TASK_VARIABILITY)
    ut_a = UserTask(env, policy, "Quick Check", SERVICE_INTERVAL, TASK_VARIABILITY)
    ut_b = UserTask(env, policy, "Relevance Test", SERVICE_INTERVAL, TASK_VARIABILITY)
    ut_c = UserTask(env, policy, "Acquisition SCORING", SERVICE_INTERVAL, TASK_VARIABILITY)
    ut_d = UserTask(env, policy, "Relevance Test 2", SERVICE_INTERVAL, TASK_VARIABILITY)
    ut_e = UserTask(env, policy, "Acquisition Business Plan", 5 * SERVICE_INTERVAL, TASK_VARIABILITY)
    ut_f = UserTask(env, policy, "Review", SERVICE_INTERVAL, TASK_VARIABILITY)
    ut_g = UserTask(env, policy, "Assign Acquisition Offer", SERVICE_INTERVAL, TASK_VARIABILITY, terminal=True)
    ut_h = UserTask(env, policy, "Rejection Letter to Broker", SERVICE_INTERVAL, TASK_VARIABILITY, terminal=True)

    xor = XOR(env, "xor")
    xor_a = XOR(env, "xor_a")
    xor_b = XOR(env, "xor_b")
    cor_g = COR(env, "cor_g")
    dor_c = DOR(env, "dor_c")
    xor_d = XOR(env, "xor_d")
    xor_e = XOR(env, "xor_e")
    xor_f = XOR(env, "xor_f")
    xor_h = XOR(env, "xor_h")
    xor_i = XOR(env, "xor_i")

    ut.assign_child(xor)

    xor.assign_child(ut_a, xor_b)

    ut_a.assign_child(xor_a)

    xor_a.assign_child(ut_b, xor_b)

    xor_b.assign_child(xor_h)

    ut_b.assign_child(dor_c)

    dor_c.assign_child(cor_g, ut_c, xor_f)

    ut_c.assign_child(xor_d)

    xor_d.assign_child(ut_d, cor_g)

    ut_d.assign_child(xor_e)

    xor_e.assign_child(cor_g, xor_f)

    xor_f.assign_child(ut_e)

    ut_e.assign_child(cor_g)

    cor_g.assign_child(ut_f, xor_i, xor_h)

    ut_f.assign_child(xor_i)

    xor_i.assign_child(ut_g)

    xor_h.assign_child(ut_h)

    actions_pool = [{xor.node_id: 1, xor_b.node_id: 0, xor_h.node_id: 0},
                    {xor.node_id: 0, xor_a.node_id: 1, xor_b.node_id: 0, xor_h.node_id: 0},
                    {xor.node_id: 0, xor_a.node_id: 0, dor_c.node_id: 0, cor_g.node_id: 0, xor_i.node_id: 0},
                    {xor.node_id: 0, xor_a.node_id: 0, dor_c.node_id: 0, cor_g.node_id: 0, xor_i.node_id: 0},
                    {xor.node_id: 0, xor_a.node_id: 0, dor_c.node_id: 0, cor_g.node_id: 1, xor_i.node_id: 0},
                    {xor.node_id: 0, xor_a.node_id: 0, dor_c.node_id: 0, cor_g.node_id: 2, xor_h.node_id: 0},
                    {xor.node_id: 0, xor_a.node_id: 0, dor_c.node_id: (1, 2), xor_d.node_id: 1, xor_f.node_id: 0,
                     cor_g.node_id: 2,
                     xor_h.node_id: 0},
                    {xor.node_id: 0, xor_a.node_id: 0, dor_c.node_id: (1, 2), xor_d.node_id: 0, xor_f.node_id: 0,
                     xor_e.node_id: 0,
                     cor_g.node_id: 2,
                     xor_h.node_id: 0}
                    ]

    weights = [1 / len(actions_pool) for _ in range(len(actions_pool))]

    master_state = pcg.RandomState(seed)

    se = StartEvent(env, generation_interval, actions_pool, weights, master_state, accelerate, starting_generation,
                    sim_time, sigmoid_param)
    se.assign_child(ut)

    return se


def simple_process(env, policy, seed, generation_interval, accelerate, starting_generation, sim_time, sigmoid_param):
    ut = UserTask(env, policy, "User Task", SERVICE_INTERVAL, TASK_VARIABILITY, terminal=True)
    master_state = pcg.RandomState(seed)

    se = StartEvent(env, generation_interval, None, None, master_state, accelerate, starting_generation,
                    sim_time, sigmoid_param)
    se.assign_child(ut)

    return se
