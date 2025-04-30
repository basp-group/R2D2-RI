import logging

def vprint(message, verbose):
    if verbose:
        print(message, flush=True)

def remove_lightning_console_log():
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
