# "global" variables
AVG_SYS_LOAD = 0.5
NUMBER_OF_USERS = 2
SERVICE_INTERVAL = 1
LAMBDA_ARR = AVG_SYS_LOAD * NUMBER_OF_USERS / SERVICE_INTERVAL
GENERATION_INTERVAL = 1 / LAMBDA_ARR
SIM_TIME = 100
BATCH_SIZE = 3
TASK_VARIABILITY = 0.2*SERVICE_INTERVAL
WORKER_VARAIBILITY = 0.2*SERVICE_INTERVAL


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
