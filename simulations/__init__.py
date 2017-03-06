from elements.workflow_process_elements import StartEvent, UserTask, XOR,DOR,COR, connect

# "global" variables
AVG_SYS_LOAD = 0.1
NUMBER_OF_USERS = 2
SERVICE_INTERVAL = 1
LAMBDA_ARR = AVG_SYS_LOAD * NUMBER_OF_USERS / SERVICE_INTERVAL
GENERATION_INTERVAL = 1 / LAMBDA_ARR
SIM_TIME = 100
BATCH_SIZE = 2
TASK_VARIABILITY = 0.2 * SERVICE_INTERVAL
WORKER_VARAIBILITY = 0.2 * SERVICE_INTERVAL


def create_files(name):
    """
Uses the passed string name to initialize the required files for the analysis. Returns two file objects and two file names.
    :param name: string passed from the specific simulation script.
    :return: two file objects and their respective file names including the extension.
    """
    file_policy_name = "{}.csv".format(name)
    file_statistics_name = "{}_evolution.csv".format(name)
    file_policy = open(file_policy_name, "w")
    file_statistics = open(file_statistics_name, "w")
    file_policy.write("job,arrival,started,finished,user,task")
    file_statistics.write("now,global")
    for i in range(NUMBER_OF_USERS):
        file_policy.write(",user_{}_st".format(i + 1))
        file_statistics.write(",user_{}".format(i + 1))
    file_statistics.write(",task")
    file_policy.write("\n")
    file_statistics.write("\n")

    return file_policy, file_statistics, file_policy_name, file_statistics_name

def initialize_process(env,policy):
    actions_pool = [[1,0,0],[0,1,0,0]]

    # start event
    se = StartEvent(env, 2.5)

    # user tasks
    ut = UserTask(env, policy, "Setup Acquisition Offer", SERVICE_INTERVAL, TASK_VARIABILITY)
    ut_a = UserTask(env, policy, "Quick Check", SERVICE_INTERVAL, TASK_VARIABILITY)
    ut_b = UserTask(env, policy, "Relevance Test", SERVICE_INTERVAL, TASK_VARIABILITY)
    ut_c = UserTask(env, policy, "Acquisition SCORING", SERVICE_INTERVAL, TASK_VARIABILITY)
    ut_d = UserTask(env, policy, "Relevance Test 2", SERVICE_INTERVAL, TASK_VARIABILITY)
    ut_e = UserTask(env, policy, "Acquisition Business Plan", SERVICE_INTERVAL, TASK_VARIABILITY)
    ut_f = UserTask(env, policy, "Review", SERVICE_INTERVAL, TASK_VARIABILITY)
    ut_g = UserTask(env, policy, "Assign Acquisition Offer", SERVICE_INTERVAL, TASK_VARIABILITY,terminal=True)
    ut_h = UserTask(env, policy, "Rejection Letter to Broker", SERVICE_INTERVAL, TASK_VARIABILITY,terminal=True)

    # decision nodes
    xor = XOR(env,"xor")
    xor_a = XOR(env,"xor_a")
    xor_b = XOR(env,"xor_b")
    cor_g = COR(env,"cor_g")
    dor_c = DOR(env,"dor_c",cor_g)
    xor_d = XOR(env,"xor_d")
    xor_e = XOR(env,"xor_e")
    xor_f = XOR(env,"xor_f")
    xor_h = XOR(env,"xor_h")
    xor_i = XOR(env,"xor_i")

    # connections
    connect(se, ut)
    connect(ut,xor)
    xor.assign_children((ut_a,xor_b))
    connect(ut_a,xor_a)
    xor_a.assign_children((ut_b,xor_b))
    xor_b.children.append(xor_h)
    connect(ut_b,dor_c)
    dor_c.children.extend((cor_g,ut_c,xor_f))
    connect(ut_c,xor_d)
    xor_d.assign_children((ut_d,cor_g))
    connect(ut_d,xor_e)
    xor_e.assign_children((cor_g,xor_f))
    xor_f.children.append(ut_e)
    connect(ut_e,cor_g)
    cor_g.children.extend((ut_f,xor_i,xor_h))
    connect(ut_f,xor_i)
    xor_i.children.append(ut_g)
    xor_h.children.append(ut_h)


    return se
