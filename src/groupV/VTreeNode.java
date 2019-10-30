package groupV;

import core.GameState;
import players.heuristics.*;
import utils.ElapsedCpuTimer;
import utils.Types;
import utils.Utils;
import utils.Vector2d;

import java.util.ArrayList;
import java.util.Random;

public class VTreeNode {
	public VParams params;

	private VTreeNode parent;
	private VTreeNode[] children;
	private double totValue;
	private int nVisits;
	private Random m_rnd;
	private int m_depth;
	private double[] bounds = new double[] { Double.MAX_VALUE, -Double.MAX_VALUE };
	private int childIdx;
	private int fmCallsCount;

	private int num_actions;
	private Types.ACTIONS[] actions;

	private GameState rootState;
	private StateHeuristic rootStateHeuristic; // heuristic used in rollOut at the end of simulation

	VTreeNode(VParams p, Random rnd, int num_actions, Types.ACTIONS[] actions) {
		this(p, null, -1, rnd, num_actions, actions, 0, null);
	}

	private VTreeNode(VParams p, VTreeNode parent, int childIdx, Random rnd, int num_actions, Types.ACTIONS[] actions,
			int fmCallsCount, StateHeuristic sh) {
		this.params = p;
		this.fmCallsCount = fmCallsCount;
		this.parent = parent;
		this.m_rnd = rnd;
		this.num_actions = num_actions;
		this.actions = actions;
		children = new VTreeNode[num_actions];
		totValue = 0.0;
		this.childIdx = childIdx;
		if (parent != null) {
			m_depth = parent.m_depth + 1;
			this.rootStateHeuristic = sh;
		} else
			m_depth = 0;
	}

	void setRootGameState(GameState gs) {
		this.rootState = gs;
		
	    if (params.heuristic_method == params.CUSTOM_HEURISTIC)
	    	this.rootStateHeuristic = new CustomHeuristic(gs); 
	    else if (params.heuristic_method == params.ADVANCED_HEURISTIC) // New method: combined heuristics 
	    	this.rootStateHeuristic = new AdvancedHeuristic(gs, m_rnd);
	    else if (params.heuristic_method == params.V_HEURISTIC)  // our designed heuristic
	    	this.rootStateHeuristic = new VHeuristic(gs, m_rnd);
		 
	}

	void mctsSearch(ElapsedCpuTimer elapsedTimer) {

		double avgTimeTaken;
		double acumTimeTaken = 0;
		long remaining;
		int numIters = 0;

		int remainingLimit = 5;
		boolean stop = false;

		while (!stop) {

			// Start from a copy of the game state
			GameState state = rootState.copy();
			ElapsedCpuTimer elapsedTimerIteration = new ElapsedCpuTimer();

			// 1. Selection and 2. Expansion are executed in treePolicy(state)
			VTreeNode selected = treePolicy(state);
			// 3. Simulation - rollout
			double delta = selected.rollOut(state);
			// 4. Back-propagation
			backUp(selected, delta);

			// Stopping condition
			// Stopping condition: it can be time, number of iterations or uses of the
			// forward model.
			// For each case, update counts to determine if we must stop.
			if (params.stop_type == params.STOP_TIME) {
				numIters++;
				acumTimeTaken += (elapsedTimerIteration.elapsedMillis());
				avgTimeTaken = acumTimeTaken / numIters;
				remaining = elapsedTimer.remainingTimeMillis();
				stop = remaining <= 2 * avgTimeTaken || remaining <= remainingLimit;
			} else if (params.stop_type == params.STOP_ITERATIONS) {
				numIters++;
				stop = numIters >= params.num_iterations;
			} else if (params.stop_type == params.STOP_FMCALLS) {
				fmCallsCount += params.rollout_depth;
				stop = (fmCallsCount + params.rollout_depth) > params.num_fmcalls;
			}
		}
		// System.out.println(" ITERS " + numIters);
	}

	/**
	 * Performs the tree policy. Navigates down the tree selecting nodes using UCT,
	 * until a not-fully expanded node is reached. Then, it starts the call to
	 * expand it.
	 * 
	 * @param state Current state to do the policy from.
	 * @return the expanded node.
	 */
	private VTreeNode treePolicy(GameState state) {

		// 'cur': our current node in the tree.
		VTreeNode cur = this;

		// We keep going down the tree as long as the game is not over and we haven't
		// reached the maximum depth
		while (!state.isTerminal() && cur.m_depth < params.rollout_depth) {
			// If not fully expanded, expand this one.
			if (cur.notFullyExpanded()) {
				// This one is the node to start the rollout from.
				return cur.expand(state);
			} else {
				// If fully expanded, apply UCT to pick one of the children of 'cur'
				cur = cur.uct(state);
			}
		}
		// This one is the node to start the rollout from.
		return cur;
	}

	/**
	 * Performs the expansion phase of the MCTS iteration.
	 * 
	 * @param state Game state *before* the expansion happens (i.e. parent node that
	 *              is not fully expanded).
	 * @return The newly expande tree node.
	 */
	private VTreeNode expand(GameState state) {

		// Go through all the not-expanded children of this node and pick one at random.
		int bestAction = 0;
		double bestValue = -1;

		for (int i = 0; i < children.length; i++) {
			double x = m_rnd.nextDouble();
			if (x > bestValue && children[i] == null) {
				bestAction = i;
				bestValue = x;
			}
		}

		// Roll the state forward, using the Forward Model, applying the action chosen
		// at random
		roll(state, actions[bestAction]);

		// state is now the next state, of the expanded node. Create a node with such
		// state
		// and add it to the tree, as child of 'this'
		VTreeNode tn = new VTreeNode(params, this, bestAction, this.m_rnd, num_actions, actions, fmCallsCount,
				rootStateHeuristic);
		children[bestAction] = tn;

		// Get the expanded node back.
		return tn;
	}

	/**
	 * Uses the Forward Model to advance the state 'gs' with the action 'act'.
	 * 
	 * @param gs  Game state to advance.
	 * @param act Action to use to advance the game state.
	 */
	// OPPONENT MODELLING
	private void roll(GameState gs, Types.ACTIONS act) {
		// Simple, all random first, then my position.
		// To roll the state forward, we need to pass an action for *all* players.
		int nPlayers = 4;
		Types.ACTIONS[] actionsAll = new Types.ACTIONS[4];

		// This is the location in the array of actions according to my player ID
		int playerId = gs.getPlayerId() - Types.TILETYPE.AGENT0.getKey();

		for (int i = 0; i < nPlayers; ++i) {
			if (playerId == i) {
				// This is me, just put the action in the array.
				actionsAll[i] = act;
			} else {
				// This is another player. We can have different models:

				// Random model
				int actionIdx = m_rnd.nextInt(gs.nActions()); // Action index at random
				actionsAll[i] = Types.ACTIONS.all().get(actionIdx); // Pick the action from the array of actions
			}
		}

		// Once the array is ready, advance the state. This changes the internal 'gs'
		// object.
		gs.next(actionsAll);
	}

	/**
	 * Performs UCT in a node. Selects the action to follow during the tree policy.
	 * 
	 * @param state
	 * @return
	 */
	private VTreeNode uct(GameState state) {

		// We'll pick the action with the highest UCB1 value.
		VTreeNode selected = null;
		double bestValue = -Double.MAX_VALUE;

		// For each children, calculate the different parts.
		for (VTreeNode child : this.children) {

			double hvVal = child.totValue;
			double childValue = hvVal / (child.nVisits + params.epsilon);

			childValue = Utils.normalise(childValue, bounds[0], bounds[1]);

			double uctValue = childValue
					+ params.K * Math.sqrt(Math.log(this.nVisits + 1) / (child.nVisits + params.epsilon));

			uctValue = Utils.noise(uctValue, params.epsilon, this.m_rnd.nextDouble()); // break ties randomly

			// small sampleRandom numbers: break ties in unexpanded nodes
			if (uctValue > bestValue) {
				selected = child;
				bestValue = uctValue;
			}
		}

		if (selected == null) {
			throw new RuntimeException("Warning! returning null: " + bestValue + " : " + this.children.length + " "
					+ +bounds[0] + " " + bounds[1]);
		}

		// Roll the state:
		roll(state, actions[selected.childIdx]);

		return selected;
	}

	/**
	 * Performs the default policy (random rollout).
	 * 
	 * @param state State where the rollout starts.
	 * @return Returns the value of the state found at the end of the rollout.
	 */
	private double rollOut(GameState state) {
		// Keep track of the current depth - we won't be creating nodes here.
		int thisDepth = this.m_depth;

		// While the rollout shouldn't finish...
		while (!finishRollout(state, thisDepth)) {
			// Take a random (but safe) action from this state.
			int action = safeRandomAction(state);
			// Advance the state with it and add 1 to the depth count.
			roll(state, actions[action]);
			thisDepth++;
		}

		return rootStateHeuristic.evaluateState(state);
	}

	/**
	 * Takes a random action among the possible safe one.
	 * 
	 * @param state State to take the action from
	 * @return index of the action to execute.
	 */
	private int safeRandomAction(GameState state) {
		Types.TILETYPE[][] board = state.getBoard();
		ArrayList<Types.ACTIONS> actionsToTry = Types.ACTIONS.all();
		int width = board.length;
		int height = board[0].length;

		// For all actions
		while (actionsToTry.size() > 0) {

			// See where would this take me.
			int nAction = m_rnd.nextInt(actionsToTry.size());
			Types.ACTIONS act = actionsToTry.get(nAction);
			Vector2d dir = act.getDirection().toVec();

			Vector2d pos = state.getPosition();
			int x = pos.x + dir.x;
			int y = pos.y + dir.y;

			// Make sure there are no flames that would kill me there.
			if (x >= 0 && x < width && y >= 0 && y < height)
				if (board[y][x] != Types.TILETYPE.FLAMES)
					return nAction;

			actionsToTry.remove(nAction);
		}

		// If we got here, we couldn't find an action that wouldn't kill me. We can take
		// any, really.
		return m_rnd.nextInt(num_actions);
	}

	@SuppressWarnings("RedundantIfStatement")
	/**
	 * Checks if a rollout should finish.
	 * 
	 * @param rollerState State being rolled
	 * @param depth       How far should we go rolling it forward.
	 * @return False when we shoud continue, true if we should finish.
	 */
	private boolean finishRollout(GameState rollerState, int depth) {
		// max depth
		if (depth >= params.rollout_depth)
			return true;

		// end of game
		if (rollerState.isTerminal())
			return true;

		return false;
	}

	/**
	 * Back propagation step of MCTS. Takes the value of a state and updates the
	 * accummulated reward on each traversed node, using the parent link. Updates
	 * count visits as well. Also updates bounds of the rewards seen so far.
	 * 
	 * @param node   Node to start backup from. This node should be the one expanded
	 *               in this iteration.
	 * @param result Reward to back-propagate
	 */
	private void backUp(VTreeNode node, double result) {
		VTreeNode n = node;

		// Go up until n == null, which happens after updating the root.
		while (n != null) {
			n.nVisits++; // Another visit to this node (N(s)++)
			n.totValue += result; // Accummulate result (computationally cheaper than having a running average).

			// Update the bounds.
			if (result < n.bounds[0]) {
				n.bounds[0] = result;
			}
			if (result > n.bounds[1]) {
				n.bounds[1] = result;
			}

			// Next one, the parent.
			n = n.parent;
		}
	}

	/**
	 * Checks which one is the index of the most used action of this node. Used for
	 * the recommendation policy.
	 * 
	 * @return Returns the index of the most visited action action
	 */
	int mostVisitedAction() {
		int selected = -1;
		double bestValue = -Double.MAX_VALUE;
		boolean allEqual = true;
		double first = -1;

		for (int i = 0; i < children.length; i++) {

			if (children[i] != null) {
				if (first == -1)
					first = children[i].nVisits;
				else if (first != children[i].nVisits) {
					allEqual = false;
				}

				double childValue = children[i].nVisits;
				// As with UCT, we add small random noise to break potential ties.
				childValue = Utils.noise(childValue, params.epsilon, this.m_rnd.nextDouble()); // break ties randomly
				if (childValue > bestValue) {
					bestValue = childValue;
					selected = i;
				}
			}
		}

		// This should happen, just pick the first action available
		if (selected == -1) {
			selected = 0;
		} else if (allEqual) {
			// If all are equal (rare), we opt to choose for the one with the highest UCB1
			// value
			selected = bestAction();
		}

		return selected;
	}

	/**
	 * Returns the index of the action with the highest UCB1 value to take from this
	 * node.
	 * 
	 * @return the index of the best action
	 */
	private int bestAction() {
		int selected = -1;
		double bestValue = -Double.MAX_VALUE;

		for (int i = 0; i < children.length; i++) {

			if (children[i] != null) {
				double childValue = children[i].totValue / (children[i].nVisits + params.epsilon);
				childValue = Utils.noise(childValue, params.epsilon, this.m_rnd.nextDouble()); // break ties randomly
				if (childValue > bestValue) {
					bestValue = childValue;
					selected = i;
				}
			}
		}

		if (selected == -1) {
			System.out.println("Unexpected selection!");
			selected = 0;
		}

		return selected;
	}

	/**
	 * Checks if this node is not fully expanded. If any of my children is null,
	 * then it's not fully expanded.
	 * 
	 * @return true if the node is not fully expanded.
	 */
	private boolean notFullyExpanded() {
		for (VTreeNode tn : children) {
			if (tn == null) {
				return true;
			}
		}

		return false;
	}
}
