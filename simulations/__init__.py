from elements.workflow_process_elements import StartEvent, UserTask, connect

# "global" variables
AVG_SYS_LOAD = 0.5
NUMBER_OF_USERS = 2
SERVICE_INTERVAL = 1
LAMBDA_ARR = AVG_SYS_LOAD * NUMBER_OF_USERS / SERVICE_INTERVAL
GENERATION_INTERVAL = 1 / LAMBDA_ARR
SIM_TIME = 1000
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
    file_policy.write("job,arrival,started,finished,user")
    file_statistics.write("now,global")
    for i in range(NUMBER_OF_USERS):
        file_policy.write(",user_{}_st".format(i + 1))
        file_statistics.write(",user_{}".format(i + 1))
    file_policy.write("\n")
    file_statistics.write("\n")

    return file_policy, file_statistics, file_policy_name, file_statistics_name

def initialize_process(env,policy):
    # start event
    start_event = StartEvent(env, GENERATION_INTERVAL)

    # user tasks
    user_task = UserTask(env, policy, "User task 1", SERVICE_INTERVAL, TASK_VARIABILITY)
    user_task_two = UserTask(env, policy, "User task 2", SERVICE_INTERVAL, TASK_VARIABILITY)

    # connections
    connect(start_event, user_task)
    connect(user_task,user_task_two)

    return start_event
