import numpy as np
import itertools
import sys
import randomstate.prng.pcg64 as pcg


class Node(object):
    node_id = itertools.count()

    def __init__(self):
        self.node_id = next(Node.node_id)

    def assign_child(self, *children):
        for child in children:
            if isinstance(self, (StartEvent, UserTask)):
                self.child = child
            elif isinstance(self, (XOR, DOR, COR)):
                self.children.append(child)

    @staticmethod
    def child_forward(child, env, token):
        if isinstance(child, UserTask):
            env.process(child.claim_token(token))
        elif isinstance(child, (XOR, DOR, COR)):
            child.forward(token)


class StartEvent(Node):
    def __init__(self, env, generation_interval, actions, weights, master_state, accelerate, starting_generation,
                 sim_time, sigmoid_param):
        """
Initializes a start event object.
        :param env: simpy environment.
        :param generation_interval: mean of generation interval to be sampled.
        """
        super().__init__()
        self.env = env
        self.generation_interval = generation_interval
        self.actions = actions
        self.weights = weights
        self.master_state = master_state
        self.accelerate = accelerate
        self.starting_generation = starting_generation
        self.sim_time = sim_time
        self.sigmoid_param = sigmoid_param
        self.child = None
        self.t = []
        self.g = []

    def generate_tokens(self):
        """
Generates infinitely many tokens (implicit objects) following an exponential rate.
        """
        while True:
            random_state = self.advance_master_state()
            token = Token(random_state)
            if self.accelerate:
                exp_arrival = self.sigmoid(self.env.now)
            else:
                exp_arrival = token.random_state.exponential(self.generation_interval)
            yield self.env.timeout(exp_arrival)
            path = token.random_state.choice(self.actions, p=self.weights)
            token.actions = path
            if self.child is None:
                print("Start event has no child assigned")
                break
            else:
                self.child_forward(self.child, self.env, token)

    def sigmoid(self, t):
        sig = (self.starting_generation - self.generation_interval) / (
            1 + np.exp(self.sigmoid_param * (t - self.sim_time / 2))) + self.generation_interval
        self.t.append(t)
        self.g.append(sig)
        return sig

    def advance_master_state(self):
        current_state = self.master_state.get_state()
        random_state = pcg.RandomState()
        random_state.set_state(current_state)
        self.master_state.advance(int(1e9))
        return random_state


class UserTask(Node):
    def __init__(self, env, policy, name, service_interval, task_variability, terminal=False):
        """
Initializes a user task object.
        :param env: simpy environment.
        :param policy: assigned policy to be used by the user task.
        :param name: descriptive name.
        :param service_interval: mean of service interval to be sampled.
        :param task_variability: per user task variability to be used for sampling user specific service times.
        """
        super().__init__()
        self.env = env
        self.policy = policy
        self.name = name
        self.service_interval = service_interval
        self.task_variability = task_variability
        self.terminal = terminal
        self.child = None

    def claim_token(self, token):
        """
Claims an implicit token generated by a start event, calls for a request to its policy, yields a policyjob object for its service time and finally releases it.
        """
        token.worked_by(self)
        policy_job = self.policy.request(self, token)
        service_time = yield policy_job.request_event
        yield self.env.timeout(service_time)
        self.policy.release(policy_job)
        if self.child is None and not self.terminal:
            print("{} has no child assigned".format(self.name))
            return
        else:
            self.child_forward(self.child, self.env, token)


class XOR(Node):
    def __init__(self, env, name):
        super().__init__()
        self.env = env
        self.name = name
        self.children = []

    def forward(self, token):
        token.worked_by(self)
        action = token.get_action(self)
        child = self.children[action]
        self.child_forward(child, self.env, token)


class DOR(Node):
    def __init__(self, env, name):
        super().__init__()
        self.env = env
        self.name = name
        self.children = []

    def forward(self, token):
        token.worked_by(self)
        action = token.get_action(self)
        if not isinstance(action, int):
            for a in action:
                self.choose_child(a, token)
        else:
            self.choose_child(action, token)

    def choose_child(self, action, token):
        child = self.children[action]
        token.counter.increment()
        self.child_forward(child, self.env, token)


class COR(Node):
    def __init__(self, env, name):
        super().__init__()
        self.env = env
        self.name = name
        self.children = []

    def forward(self, token):
        token.worked_by(self)
        token.counter.decrement()
        if token.counter.count == 0:
            action = token.get_action(self)
            child = self.children[action]
            self.child_forward(child, self.env, token)


class Token(object):
    token_id = itertools.count()
    def __init__(self, random_state):
        self.random_state = random_state
        self.counter = Counter()
        self.history = []
        self.actions = None
        self.id = next(Token.token_id)

    def worked_by(self, element):
        self.history.append(element.name)

    def get_action(self, element):
        try:
            action = self.actions[element.node_id]
            return action
        except KeyError:
            sys.exit("Chosen action doesn't lead to direct child, token's action path was {}. Aborting...".format(
                self.actions))


class Counter(object):
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

    def decrement(self):
        self.count -= 1
