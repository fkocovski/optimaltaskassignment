# Optimal Job Assignment in a Discrete Event Simulation Environment

Implementation of different policies for optimal role resolution based on Zeng and Zhao (2005).

Included policies:

1. Least Loaded Qualified Person (LLQP)
2. Shared Queue (SQ)
3. K-Batch
4. K-Batch-1
5. 1-Batch-1

For solving the optimization problem present in the batching policies, the [Gurobi Optimization](http://www.gurobi.com) suite was used.

For the reinforcement learning environment all algorithms are based on the new draft of Sutton and Barto (2017)'s book [2017 Version](http://incompleteideas.net/sutton/book/bookdraft2016sep.pdf).

# References

* Zeng, D. D., & Zhao, J. L. (2005). Effective role resolution in workflow management. INFORMS journal on computing, 17(3), 374-387.
* Sutton, R. S., & Barto, A. G. (1998). Reinforcement learning: An introduction (Vol. 1, No. 1). Cambridge: MIT press.
