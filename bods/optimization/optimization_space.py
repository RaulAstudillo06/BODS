import GPyOpt

class OptimizationSpace(object):
    """
    """

    def __init__(self, decision_variables, context_variables):
        """

        :param decision_space:
        :param context_space:
        """
        #self.decision_variables = decision_variables
        #self.context_variables = context_variables
        self.decision_space = GPyOpt.Design_space(decision_variables)
        self.decision_context_space = GPyOpt.Design_space(decision_variables + context_variables)
    def decision_space_dim(self):
        """
        :return:
        """
        return self.decision_space.input_dim()


    def get_decision_bounds(self):
        """
        :return:
        """
        return self.decision_space.get_bounds()
