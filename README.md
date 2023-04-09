# InfiniteMonkeyNet
A neural network in which some monkeys adjust the weights and biases. Surely, given *enough* time, they will **inevitably** find a solution that converges to the desired network, right? (https://en.wikipedia.org/wiki/Infinite_monkey_theorem)

Well, time to find out!

**IMNet_chaos.py** attempts to train a tiny network on the good ol' XOR function. You can add as many layers as you wish as large as you want by modifying the generate_network() function. (Let your machine suffer in comfort, it's not like this particular AI is going to rise up against you any time soon.)

**IMNet_guided.py** is a little different. In this one, I did help the monkeys a "little bit" by not allowing them to make changes that increase the average error from the last step - but that results in them getting stuck in every kind of local minimums that are far from the global minimum of the error. So unless you start moving towards the global maximum right away, it means you are likely going towards a local minimum and you have to start over.
