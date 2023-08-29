# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        mdp=self.mdp
        for i in range(self.iterations):
            new_value = util.Counter()
            for state in mdp.getStates():
                if not mdp.isTerminal(state):
                    new_value[state]=max(self.getQValue(state,action) for action in mdp.getPossibleActions(state))
            self.values= new_value



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        mdp= self.mdp
        total_q=0
        state_and_prob=mdp.getTransitionStatesAndProbs(state, action)
        for succ, prob in state_and_prob:
            total_q+= prob*(mdp.getReward(state,action,succ)+self.discount*self.getValue(succ))

        return total_q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        best_value = -float("inf")
        best_action=""
        if mdp.isTerminal(state):
            return None
        else:
            for action in mdp.getPossibleActions(state):
                value= self.getQValue(state,action)
                if value > best_value:
                    best_action= action
                    best_value= value
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        counter=0
        mdp = self.mdp
        states= mdp.getStates()
        n=len(states)
        for i in range(self.iterations):
            state=states[counter]
            if not mdp.isTerminal(state):
                self.values[state]=max(self.getQValue(state,action) for action in mdp.getPossibleActions(state))
            counter+=1
            if counter== n:
                counter=0

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        stack=util.PriorityQueue()
        mdp= self.mdp
        states= mdp.getStates()
        predecessor={}
        for state in states:
            if mdp.isTerminal(state):
                continue
            actions=mdp.getPossibleActions(state)
            highest_q=max(self.getQValue(state,action) for action in actions)
            diff = abs(highest_q-self.values[state])
            stack.push(state,-diff)
            for action in actions:
                succ_and_prob=mdp.getTransitionStatesAndProbs(state,action)
                for succ, prob in succ_and_prob:
                    if succ in predecessor:
                        predecessor[succ].add(state)
                    else:
                        predecessor[succ]=set([state])
        for i in range(self.iterations):
            if stack.isEmpty():
                break
            s= stack.pop()
            if not mdp.isTerminal(s):
                actions = mdp.getPossibleActions(s)
                self.values[s] = max(self.getQValue(s,action) for action in actions)
            for p in predecessor[s]:
                actions=mdp.getPossibleActions(p)
                highest_q = max(self.getQValue(p,action) for action in actions)
                diff = abs(highest_q - self.values[p])
                if diff > self.theta:
                    stack.update(p, -diff)


