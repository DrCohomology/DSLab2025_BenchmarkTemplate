import src.config as cfg


def collect_results(result_dir):
    """
    Opens and concatenates all of the parquet files in result_dir.
        These files should all be experimental results.
    Then, stores the concatenated dataset in a new file called results.parquet.
        IMPORTANT: results.pqrquet should NOT be saved in result_dir
    """
    return





if __name__ == "__main__":
    collect_results(cfg.RESULT_DIR)
