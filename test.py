from scipy.stats import  kstest
from scipy.stats import lognorm


def perform_ks_test(data, distribution='norm', *args):
    """
    Perform the Kolmogorov-Smirnov test on the given data for a specified distribution.

    Parameters:
        data (array-like): The empirical data to test.
        distribution (str): The name of the theoretical distribution (e.g., 'norm', 't', 'lognorm').
        *args: Parameters for the specified distribution (e.g., mean and std for 'norm').

    Returns:
        dict: A dictionary containing the test statistic (D) and p-value.
    """
    # Ensure there are no NaN values in the data
    clean_data = data.dropna() if hasattr(data, 'dropna') else data

    # Perform the Kolmogorov-Smirnov test
    ks_result = kstest(clean_data, distribution, args=args)

    # Return the results as a dictionary
    return {
        'D': ks_result.statistic,  # KS statistic
        'p-value': ks_result.pvalue  # p-value of the test
    }