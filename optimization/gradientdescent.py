

def gradient_descent(start_thetas, cost_func, gradient_func, args=tuple(),
                     learning_rate=3,max_iter=10000,min_iter=0,diff_threshold=1e-16, cost_threshold=1e-5):
    # Initialize function arguments
    f_args = list(args)
    f_args.insert(0, start_thetas)

    def has_converged(i, max_iter, cost_history : list):
        if i < min_iter:
            return False
        if max_iter > 0 and i >= max_iter:
            return True

        l = len(cost_history)
        diff = cost_history[l-2] - cost_history[l-1]
        return abs(diff) < diff_threshold or cost_history[l-1] < cost_threshold

    # Calculate the first cost
    cost_history = [cost_func(*f_args)]
    i = 0
    converged = False
    thetas = start_thetas
    while not converged:
        i += 1
        # do something here
        thetas -= learning_rate * gradient_func(*f_args)
        f_args[0] = thetas
        cost_history.append(cost_func(*f_args))
        converged = has_converged(i, max_iter, cost_history)

    return thetas
