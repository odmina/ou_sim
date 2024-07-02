from OU_process_2v import OU_process_2v


class person(object):
    """
    This class stores parameters and data of a given "person".

    Parameters include:
    `ind_diffs` - variables that define stable individual differences.
    Can be set when an instance is createdm or later with `add_ind_diffs()`.
    Stored as a dictionary.
    `observations` - stores  data along with parameters.
    Stored as a nested dictionary, initialized empty.
    Can be set latter with `add_observations()` method.

    `simulate_OU_process_2v()` can be used to simulate data with pre-defined
    OU process.
    """

    def __init__(self, ind_diffs: dict = {}):
        if not isinstance(ind_diffs, dict):
            raise TypeError("ind_diffs should be provided as a dictionary")
        self.ind_diffs = {}  # this line is here so that add_ind_diffs() works properly
        self.add_ind_diffs(ind_diffs)
        self.observations = {}

    def __str__(self):
        return """Person data container. Use:
            `get_ind_diffs()` to access characteristics
            `get_observations()` to access observations
            """

    # ------GETTERS-------

    def get_ind_diffs(self):
        return self.ind_diffs

    def get_observations(self, set_name=None):
        if set_name is None:
            return self.observations
        else:
            return self.observations[set_name]

    def available_datasets(self):
        return self.observations.keys()

    def get_data(self, set_name):
        return self.observations[set_name]["data"]

    def get_parameters(self, set_name):
        return self.observations[set_name]["parameters"]

    # ------SETTERS-------

    def add_ind_diffs(self, added_ind_diffs={}):
        """
        This method can be used to add individual differences.

        Individual differences should be added as a dictionary
        with {'name': 'value'}. The dictionary can have >1 entry.

        Further entries can be added at a later time. If an entry
        exists, you are going to be asked if you want to overwrite it.
        If you decide not to overwrite it, NO changes will be made
        during this particular call (other entries will not be added
        and the dictionary will stay as it was before you called the
        method).
        """

        if not isinstance(added_ind_diffs, dict):
            raise TypeError("added_ind_diffs should be provided as a dictionary")

        backup = self.ind_diffs
        for entry in added_ind_diffs:
            if entry in self.ind_diffs.keys():
                cont = input(
                    entry + " already exist, type y if you want to overwrite it: ")
                if cont.strip() != "y":
                    self.ind_diffs = backup
                    del backup
                    print("No changes were made")
                    break
            self.ind_diffs[entry] = added_ind_diffs[entry]

    def add_observations(self, set_name, parameters: dict = {}, added_obs=None):
        """
        A method used to add sets of observations.
        Each set should have a name, parameters also can (and should!) be provided.
        Observations data can be added in any format.

        The dictionary that stores the observations has the following structure:
        {'set_name' :
            {
                'parameters': preferably a dictionary,
                'data': here go data in any format
            }

       You can add an empty set (no parameters, no data).
            If you add just a name (no params), you can
            add data and/or parameters later.

        Although you can add just the parameters or just the data
            with no parameters, you cannot add or change just the data
            or just the parameters of an existing dataset.
            You will have to overwrite the whole entry.
            It is not a bug, it is supposed to be this way :)
            (to prevent storing data wrong parameters).

        If you try to add a dataset with a name that already exists,
            you will be asked whether you want to overwrite it.
        """

        assert type(set_name) is str, \
            "Observations set should have a name!"

        if set_name in self.observations.keys():
            print("Dataset with this name already exists")
            cont = input("Type y if you want to overwrite it: ")
            if cont.strip() != "y":
                print("No changes were made")
                return

        self.observations[set_name] = {
            "parameters": parameters,
            "data": added_obs
        }

    def simulate_OU_process_2v(self, set_name, ou_process, mu, dt, time):
        assert type(set_name) is str, \
            "Observations set should have a name!"
        assert type(ou_process) is OU_process_2v, \
            "Provide a valid object of OU_process_2v class"

        extracted_params = {
            'mu': mu,
            'B (drift)': ou_process.get_B(),
            'Discrete B': ou_process.get_sim_discreteB(dt),
            'Gamma (stationary cov)': ou_process.get_Gamma(),
            'Sigma (diffusion)': ou_process.get_Sigma(),
            'Simulation cond. cov': ou_process.get_sim_condPDF_covariance(dt),
            'Delta t': dt,
            'Total time': time
        }
        simulated_data = ou_process.sim_data(
            mu=mu,
            d=dt,
            total_time=time)
        self.add_observations(set_name=set_name,
                              parameters=extracted_params,
                              added_obs=simulated_data)
