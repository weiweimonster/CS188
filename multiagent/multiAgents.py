# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best


        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newCapsuls=successorGameState.getCapsules()
        food_list=newFood.asList()
        # "*** YOUR CODE HERE ***"
        ghost_distance=[]
        food_distance = []
        scared_ghost_distances=[]
        food_amount=len(food_list)
        capsuls_left=len(newCapsuls)
        for ghost in newGhostStates:
            x, y = ghost.getPosition()
            if ghost.scaredTimer ==0:
                ghost_distance.append(manhattanDistance((x,y),newPos))
            else:
                scared_ghost_distances.append((manhattanDistance((x,y),newPos)))
        if scared_ghost_distances:
            scared_mindist=min(scared_ghost_distances)
        else:
            scared_mindist=0
        if ghost_distance:
            ghost_mindist=min(ghost_distance)
        else:
            ghost_mindist=float("inf")
        if not newFood.asList():
            food_mindist=0
        else:
            for food in food_list:
                food_distance.append(manhattanDistance(food,newPos))
            food_mindist=min(food_distance)
        score=successorGameState.getScore()-7/((ghost_mindist+1))-0.3*food_mindist-3*scared_mindist-4*food_amount-25*capsuls_left
        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # legalMoves = gameState.getLegalActions()
        #
        # # Choose one of the best actions
        # scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        # bestScore = max(scores)
        # bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        # "*** YOUR CODE HERE ***"
        num_agents=gameState.getNumAgents()
        def minimax(gamestate, index, depth):
            if gamestate.isWin() or gamestate.isLose() or depth==0:
                value= self.evaluationFunction(gamestate)
                return value, Directions.STOP
            elif index==0:
                value, action = max(gamestate,depth)
                return value, action
            else:
                value, action =mini(gamestate, index, depth)
                return value, action

        def mini(gamestate, index ,depth):
            actions = gamestate.getLegalActions(index)
            succ_state = [(gamestate.generateSuccessor(index, action), action) for action in actions]
            if index==num_agents-1:
                value_action_pairs = [(minimax(succ[0], 0, depth-1)[0], succ[1]) for succ in succ_state]
            else:
                value_action_pairs = [(minimax(succ[0], index+1, depth)[0], succ[1]) for succ in succ_state]
            worst_value = float("inf")
            worst_action = None
            for v, a in value_action_pairs:
                if v < worst_value:
                    worst_value = v
                    worst_action = a
            return worst_value, worst_action
        def max(gamestate,depth):
            actions=gamestate.getLegalActions(0)
            succ_state=[(gamestate.generateSuccessor(0, action),action) for action in actions]
            value_action_pairs=[(minimax(succ[0], 1 , depth)[0], succ[1]) for succ in succ_state]
            best_value=-float("inf")
            best_action=None
            for v, a in value_action_pairs:
                if v>best_value:
                    best_value=v
                    best_action=a
            return best_value, best_action
        return minimax(gameState, 0, self.depth)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        num_agents = gameState.getNumAgents()
        def minimax_ab(gamestate, index, depth, a1, b):
            if gamestate.isWin() or gamestate.isLose() or depth == 0:
                value = self.evaluationFunction(gamestate)
                return value, Directions.STOP
            elif index == 0:
                value, action = max_ab(gamestate, depth,a1 , b)
                return value, action
            else:
                value, action = mini_ab(gamestate, index, depth, a1, b)
                return value, action

        def mini_ab(gamestate, index, depth, a1, b):
            worst_value= float("inf")
            worst_action=None
            actions = gamestate.getLegalActions(index)
            for action in actions:
                succ= (gamestate.generateSuccessor(index, action), action)
                if index== num_agents-1:
                    value_action_pair=(minimax_ab(succ[0], 0, depth - 1, a1, b)[0], succ[1])
                else:
                    value_action_pair=(minimax_ab(succ[0], index + 1, depth, a1, b)[0], succ[1])
                if value_action_pair[0]<worst_value:
                    worst_value=value_action_pair[0]
                    worst_action=value_action_pair[1]
                if worst_value< a1:
                    return worst_value, worst_action
                b= min(b, worst_value)
            return worst_value, worst_action
            # succ_state = [(gamestate.generateSuccessor(index, action), action) for action in actions]
            # if index == num_agents - 1:
            #     value_action_pairs = [(minimax_ab(succ[0], 0, depth - 1, a1, b)[0], succ[1]) for succ in succ_state]
            # else:
            #     value_action_pairs = [(minimax_ab(succ[0], index + 1, depth, a1, b)[0], succ[1]) for succ in succ_state]
            # worst_value = float("inf")
            # worst_action = None
            # for v, a in value_action_pairs:
            #     if v < worst_value:
            #         worst_value = v
            #         worst_action = a
            #     if worst_value< a1:
            #         return worst_value, worst_action
            #     b= min(b, worst_value)
            # return worst_value, worst_action

        def max_ab(gamestate, depth, a1, b):
            actions = gamestate.getLegalActions(0)
            best_value = -float("inf")
            best_action = None
            actions = gamestate.getLegalActions(0)
            for action in actions:
                succ = (gamestate.generateSuccessor(0, action), action)
                value_action_pair = (minimax_ab(succ[0], 1, depth, a1, b)[0], succ[1])
                if value_action_pair[0] > best_value:
                    best_value = value_action_pair[0]
                    best_action = value_action_pair[1]
                if best_value > b:
                    return best_value, best_action
                a1 = max(a1, best_value)
            return best_value, best_action
            # succ_state = [(gamestate.generateSuccessor(0, action), action) for action in actions]
            # value_action_pairs = [(minimax_ab(succ[0], 1, depth, a1, b)[0], succ[1]) for succ in succ_state]
            # best_value = -float("inf")
            # best_action = None
            # for v, a in value_action_pairs:
            #     if v > best_value:
            #         best_value = v
            #         best_action = a
            #     if best_value>= b:
            #         return best_value, best_action
            #     a1= max(a1,best_value)
            # return best_value, best_action
        return minimax_ab(gameState, 0, self.depth,-float("inf"), float("inf"))[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        num_agents = gameState.getNumAgents()

        def expectimax(gamestate, index, depth):
            if gamestate.isWin() or gamestate.isLose() or depth == 0:
                value = self.evaluationFunction(gamestate)
                return value, Directions.STOP
            elif index == 0:
                value, action = max(gamestate, depth)
                return value, action
            else:
                value, action = expecti(gamestate, index, depth)
                return value, action

        def expecti(gamestate, index, depth):
            actions = gamestate.getLegalActions(index)
            succ_state = [(gamestate.generateSuccessor(index, action), action) for action in actions]
            if index == num_agents - 1:
                value = [expectimax(succ[0], 0, depth - 1)[0] for succ in succ_state]
            else:
                value= [expectimax(succ[0], index + 1, depth)[0] for succ in succ_state]
            average_value=sum(value)/len(value)
            return average_value, None

        def max(gamestate, depth):
            actions = gamestate.getLegalActions(0)
            succ_state = [(gamestate.generateSuccessor(0, action), action) for action in actions]
            value_action_pairs = [(expectimax(succ[0], 1, depth)[0], succ[1]) for succ in succ_state]
            best_value = -float("inf")
            best_action = None
            for v, a in value_action_pairs:
                if v > best_value:
                    best_value = v
                    best_action = a
            return best_value, best_action

        return expectimax(gameState, 0, self.depth)[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I just modified what I have in q1. but since we are evaluation states we calculate the score from
    current state instead of from successor state. This coefficient of the components indicates what will dictate
    the movement of the pacman.the most important thing of course is to not die, so the distance to the closest
    ghost will be a major concern. One of another important factor is the amount of the food left, so that would
    also be a factor. Last but not least, the amount of capsules left could also be a important factor.if the ghosts
    are scared, pacman can move freely without trying to avoid the ghost, and thus increasing our score. Also, I want
    pacman to chase after scared ghost so they will be reset and thus buy us more time to collect food
    """
    "*** YOUR CODE HERE ***"
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    Capsuls = currentGameState.getCapsules()
    food_list = Food.asList()
    # "*** YOUR CODE HERE ***"
    ghost_distance = []
    food_distance = []
    scared_ghost_distances = []
    food_amount = len(food_list)
    capsuls_left = len(Capsuls)
    for ghost in GhostStates:
        x, y = ghost.getPosition()
        if ghost.scaredTimer == 0:
            ghost_distance.append(manhattanDistance((x, y), Pos))
        else:
            scared_ghost_distances.append((manhattanDistance((x, y), Pos)))
    if scared_ghost_distances:
        scared_mindist = min(scared_ghost_distances)
    else:
        scared_mindist = 0
    if ghost_distance:
        ghost_mindist = min(ghost_distance)
    else:
        ghost_mindist = float("inf")
    if not Food.asList():
        food_mindist = 0
    else:
        for food in food_list:
            food_distance.append(manhattanDistance(food, Pos))
        food_mindist = min(food_distance)
    score = currentGameState.getScore() - 7 / (
    (ghost_mindist + 1)) - food_mindist - 3 * scared_mindist - 4 * food_amount - 25 * capsuls_left
    return score

# Abbreviation
better = betterEvaluationFunction
