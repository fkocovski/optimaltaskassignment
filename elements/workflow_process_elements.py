import numpy as np


class StartEvent(object):
    def __init__(self, env, generation_interval):
        self.env = env
        self.generation_interval = generation_interval
        self.child = None
        self.RANDOM_STATE_ARRIVAL = np.random.RandomState(1)

    def generate_tokens(self):
        """
Generates infinitely many tokens (implicit objects) following an exponential rate.
        """
        while True:
            exp_arrival = self.RANDOM_STATE_ARRIVAL.exponential(self.generation_interval)
            yield self.env.timeout(exp_arrival)
            if self.child is None:
                print("Start event has no child assigned")
                break
            elif isinstance(self.child, UserTask):
                self.env.process(self.child.claim_token())
            else:
                object_type = type(self.child)
                print("Wrong child type {}".format(object_type))
                break


class UserTask(object):
    def __init__(self, env, policy, name, service_interval, task_variability):
        self.env = env
        self.policy = policy
        self.name = name
        self.service_interval = service_interval
        self.task_variability = task_variability
        self.child = None

    def claim_token(self):
        """
Claims an implicit token generated by a start event, calls for a request to its policy, yields a policyjob object for its service time and finally releases it.
        """
        policy_job = self.policy.request(self)
        service_time = yield policy_job.request_event
        yield self.env.timeout(service_time)
        self.policy.release(policy_job)


def connect(source, destination):
    """
Establishes a parent-child relationship between source and destination.
    :param source: parent.
    :param destination: child.
    """
    source.child = destination
