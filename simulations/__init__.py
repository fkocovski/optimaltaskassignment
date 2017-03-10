from elements.workflow_process_elements import StartEvent, UserTask, XOR, DOR, COR, connect

NUMBER_OF_USERS = 2
SERVICE_INTERVAL = 1
GENERATION_INTERVAL = 5
SIM_TIME = 100
BATCH_SIZE = 2
TASK_VARIABILITY = 0.2 * SERVICE_INTERVAL
WORKER_VARIABILITY = 0.2 * SERVICE_INTERVAL


def create_files(name):
    file_policy = open(name, "w")
    file_policy.write("job,arrival,assigned,started,finished,user,task,task_name")
    for i in range(NUMBER_OF_USERS):
        file_policy.write(",user_{}_st".format(i + 1))
    file_policy.write("\n")
    return file_policy


def acquisition_process(env, policy):
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

    connect(ut, xor)
    xor.assign_children((ut_a, xor_b))
    connect(ut_a, xor_a)
    xor_a.assign_children((ut_b, xor_b))
    xor_b.children.append(xor_h)
    connect(ut_b, dor_c)
    dor_c.children.extend((cor_g, ut_c, xor_f))
    connect(ut_c, xor_d)
    xor_d.assign_children((ut_d, cor_g))
    connect(ut_d, xor_e)
    xor_e.assign_children((cor_g, xor_f))
    xor_f.children.append(ut_e)
    connect(ut_e, cor_g)
    cor_g.children.extend((ut_f, xor_i, xor_h))
    connect(ut_f, xor_i)
    xor_i.children.append(ut_g)
    xor_h.children.append(ut_h)

    actions_pool = [{xor.node_id: 1, xor_b.node_id: 0, xor_h.node_id: 0},
                    {xor.node_id: 0, xor_a.node_id: 1,xor_b.node_id:0, xor_h.node_id: 0},
                    {xor.node_id: 0, xor_a.node_id: 0,dor_c.node_id:0, cor_g.node_id:0,xor_i.node_id:0},
                    {xor.node_id: 0, xor_a.node_id: 0, dor_c.node_id: 0, cor_g.node_id: 0, xor_i.node_id: 0},
                    {xor.node_id: 0, xor_a.node_id: 0, dor_c.node_id: 0, cor_g.node_id: 1, xor_i.node_id: 0},
                    {xor.node_id: 0, xor_a.node_id: 0, dor_c.node_id: 0, cor_g.node_id: 2, xor_h.node_id: 0},
                    {xor.node_id: 0, xor_a.node_id: 0, dor_c.node_id: (1, 2), xor_d.node_id: 1, xor_f.node_id: 0, cor_g.node_id: 2,
                     xor_h.node_id: 0},
                    {xor.node_id: 0, xor_a.node_id: 0, dor_c.node_id: (1, 2), xor_d.node_id: 0, xor_f.node_id: 0,xor_e.node_id:0,
                     cor_g.node_id: 2,
                     xor_h.node_id: 0}
                    ]
    weights = [1/len(actions_pool) for _ in range(len(actions_pool))]

    se = StartEvent(env, GENERATION_INTERVAL, actions_pool, weights)
    connect(se, ut)

    return se
